import sys
import os
import torch
from torch import optim
import time
from torch.backends import cudnn
import yaml
from loss import loss
from model import HasNet
from Utils.utils import *
from Utils.dataLoader import *
from Utils.Parser import *


class trainer:
    def __init__(self, config, dataloader, criterion, model):
        self.initialize(config)
        self.dataloader = dataloader
        self.lossFunction = criterion
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            self.lossFunction = criterion(device=self.device)

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
        self.choiceNum = config['choiceNum']
        self.choiceLayer = config['choiceLayer']

    def train(self, exp_no=0, useSLC=False):
        print(f'Using device {self.device}')
        self.model.to(device=self.device)

        # self.model.to(device=self.device)
        # cudnn.benchmark = True

        optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)
        scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

        try:
            trainLoss = []
            trainReconLoss = []
            trainRLoss = []
            trainEdgeLoss = []
            trainVggLoss = []

            latencyThresholds = [100, 10, 9, 8, 7.5]
            binSize = self.epochs // len(latencyThresholds)

            for iter in range(self.epochs):
                trainLossSum = 0
                trainReconLossSum = 0
                trainRLossSum = 0
                trainEdgeLossSum = 0
                trainVggLossSum = 0

                idx = 0
                hookNum = -1
                startTime = time.time()

                for LLow, LHigh, name in self.dataloader:
                    arch = [np.random.randint(self.choiceNum) for _ in range(self.choiceLayer)]

                    # calculate the latency
                    if useSLC is True:
                        archLatency = getLatency(arch, numChoice=self.choiceNum, numLayer=self.choiceLayer)
                        archLatencyThresholds = latencyThresholds[max(iter // binSize, len(latencyThresholds) - 1)]
                        while archLatency > archLatencyThresholds and np.random.randn(1) > 0.3:
                            arch = [np.random.randint(self.choiceNum) for _ in range(self.choiceLayer)]
                            archLatency = getLatency(arch)
                    #arch = [1, 1, 1, 1, 1, 1, 1, 1]
                    #print(arch)
                    LLow = LLow.to(self.device)
                    LHigh = LHigh.to(self.device)
                    RLow, ILow = self.model(LLow, arch=arch)
                    RHigh, IHigh = self.model(LHigh, arch=arch)

                    loss, lossComponents = self.lossFunction(RLow, RHigh, ILow, IHigh, LLow, LHigh)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if idx % 8 == 0:
                        print(f"iter: {iter}_{idx}\t average_loss: {loss.item():.6f}")
                    idx += 1

                    # compute loss
                    trainLossSum += loss
                    trainReconLossSum += lossComponents['ReconLoss'].item()
                    trainRLossSum += lossComponents['RLoss'].item()
                    trainEdgeLossSum += lossComponents['EdgeLoss'].item()
                    trainVggLossSum += lossComponents['VggLoss'].item()

                # save loss
                trainLoss.append(trainLossSum)
                trainReconLoss.append(trainReconLossSum)
                trainRLoss.append(trainRLossSum)
                trainEdgeLoss.append(trainEdgeLossSum)
                trainVggLoss.append(trainVggLossSum)

                # save weight
                if iter % self.saveFrequency:
                    if not os.path.exists('./weights/{}'.format(exp_no)):
                        os.mkdir('./weights/{}'.format(exp_no))
                    if not os.path.exists('./weights/{}/supernet'.format(exp_no)):
                        os.mkdir('./weights/{}/supernet'.format(exp_no))
                    if not os.path.exists('./weights/{}/supernet/model'.format(exp_no)):
                        os.mkdir('./weights/{}/supernet/model'.format(exp_no))

                    torch.save(self.model.state_dict(),
                               './weights/{}/supernet/model/checkpoint-{}.pth'.format(exp_no, iter))
                    torch.save(self.model.state_dict(),
                               './weights/{}/supernet/model/checkpoint-latest.pth'.format(exp_no))

        except KeyboardInterrupt:
            torch.save(self.model.state_dict(), 'INTERRUPTED_decom.pth')
            log('Saved interrupt decom')
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


if __name__ == "__main__":
    criterion = loss()
    model = HasNet()

    parser = Parser()
    args = parser.parse()
    args.checkpoint = False
    if args.checkpoint:
        pretrain = torch.load('../weights/DecomNet.pth')
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

    trainer = trainer(config, trainLoader, criterion, model)

    if args.mode == 'train':
        trainer.train()
