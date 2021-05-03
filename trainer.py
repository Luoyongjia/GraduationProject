import sys
import os
import torch
from torch import optim
import time
from torch.backends import cudnn
from torchsummary import summary
import yaml
from loss import loss
from model import mNet
from dataLoader import *
from utils import *
from Parser import Parser


class trainer:
    def __init__(self, config, dataloader, criterion, model, dataloderTest=None, extraModel=None):
        self.initialize(config)
        self.dataloader = dataloader
        self.dataloaderTest = dataloderTest
        self.loss = criterion
        self.model = model
        self.extraModel = extraModel
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True

    def initialize(self, config):
        self.bachSize = config['batchSize']
        self.length = config['length']
        self.epochs = config['epochs']
        self.stepsPerEpoch = config['stepsPerEpoch']
        self.printFrequency = config['printFrequency']
        self.saveFrequency = config['saveFrequency']
        self.weightsDir = config['weightsDir']
        self.samplesDir = config['samplesDir']
        self.learningRate = config['learningRate']
        self.noDecom = config['noDecom']

    def train(self):
        print(f'Using device {self.device}')
        self.model.to(device=self.device)
        summary(self.model, input_size=(3, 48, 48))

        # self.model.to(device=self.device)
        # cudnn.benchmark = True

        optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)
        try:
            for iter in range(self.epochs):
                epochLoss = 0
                idx = 0
                hookNumber = -1
                iterStartTime = time.time()
                for LLowTensor, LHighTensor, name in self.dataloader:
                    LLow = LLowTensor.to(self.device)
                    LHigh = LHighTensor.to(self.device)
                    RLow, ILow = self.model(LLow)
                    RHigh, IHigh = self.model(LHigh)
                    if idx % self.printFrequency == 0:
                        hookNumber = -1
                    loss = self.loss(RLow, RHigh, ILow, IHigh, LLow, LHigh, hook=hookNumber)
                    hookNumber = -1
                    if idx % 2 == 0:
                        print(f'iter:{iter + 1}_{idx}\t average loss:{loss.item():.6f}')
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()
                    idx += 1

                if (iter+1) % self.printFrequency == 0:
                    self.test(iter + 1, plotDir='./images/DecomSample')

                if (iter+1) % self.saveFrequency == 0:
                    torch.save(self.model.state_dict(), './weights/DecomNet.pth')
                    log("Weight has saved as 'DecomNet.pth'")

                scheduler.step()
                iterEndTime = time.time()
                log(f"Time taken: {iterEndTime - iterStartTime:.3f} seconds\t lr={scheduler.get_lr()[0]:.6f}")
        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'INTERRUPTED_Decom.pth')
            log('Saved interrupt decom')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)

    def test(self, epoch=-1, plotDir='/images/samples'):
        self.model.eval()
        hook = 0
        for LLowTensor, LHighTensor, name in self.dataloaderTest:
            LLow = LLowTensor.to(self.device)
            LHigh = LHighTensor.to(self.device)
            RLow, ILow = self.model(LLow)
            RHigh, IHigh = self.model(LHigh)

            if epoch % self.printFrequency == 0:
                loss = self.loss(RLow, RHigh, ILow, IHigh, LLow, LHigh, hook=hook)
                hook += 1
                loss = 0

            RLownp = RLow.detach().cpu().numpy()[0]
            RHighnp = RHigh.detach().cpu().numpy()[0]
            ILownp = ILow.detach().cpu().numpy()[0]
            IHighnp = IHigh.detach().cpu().numpy()[0]
            LLownp = LLow.detach().cpu().numpy()[0]
            LHighnp = LHigh.detach().cpu().numpy()[0]
            sampleImages = np.concatenate((RLownp, ILownp, LLownp, RHighnp, IHighnp, LHighnp), axis=0)

            filepath = os.path.join(plotDir, f'{name[0]}_epoch_{epoch}.png')
            splitPoint = [0, 3, 4, 7, 10, 11, 14]
            imgDim = ILownp.shape[1:]
            sample(sampleImages, split=splitPoint, figure_size=(2, 3), img_dim=imgDim, path=filepath, num=epoch)


if __name__ == "__main__":
    criterion = loss()
    model = mNet()

    parser = Parser()
    args = parser.parse()
    args.checkpoint = True
    if args.checkpoint:
        pretrain = torch.load('./weights/DecomNet.pth')
        model.load_state_dict(pretrain)
        print('Model laded from DecomNet.pth')

    with open(args.config) as f:
        config = yaml.load(f)

    trainPath = '/Users/luoyongjia/Research/Data/MIT/train'
    vailPath = '/Users/luoyongjia/Research/Data/MIT/validation'
    testPath = '/Users/luoyongjia/Research/Data/MIT/test'
    trainListPath = buildDatasetListTxt(trainPath)
    vailListPath = buildDatasetListTxt(vailPath)
    testListPath = buildDatasetListTxt(testPath)
    log("Building dataset...")
    BatchSize = 2
    trainData = loadDataset(trainPath, trainListPath, cropSize=config['length'], toRAM=True)
    vailData = loadDataset(vailPath, vailListPath, cropSize=config['length'], toRAM=True, training=False)
    testData = loadDataset(testPath, testListPath, cropSize=config['length'], toRAM=False, training=False)

    trainLoader = DataLoader(trainData, batch_size=config['batchSize'], shuffle=True)
    vailLoader = DataLoader(vailData, batch_size=1)
    testLoader = DataLoader(testData, batch_size=1)

    trainer = trainer(config, trainLoader, criterion, model, dataloderTest=vailLoader)

    if args.mode == 'train':
        trainer.train()
    else:
        trainer.test()
