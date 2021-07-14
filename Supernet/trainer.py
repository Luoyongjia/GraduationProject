import sys
import os
import torch
from torch import optim
import time
from torch.backends import cudnn
import yaml
from Model.model_search import HasNet
from Model.loss import loss
from Utils.utils import *
from Utils.dataLoader import *
from Utils.Parser import Parser

logger = log1('test', 'Supernet')


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
        self.epochs = config['epochs']
        self.printFrequency = config['printFrequency']
        self.saveFrequency = config['saveFrequency']
        self.weightsDir = config['weightsDir']
        self.learningRate = config['learningRate']
        self.choiceNum = config['choiceNum']
        self.choiceLayer = config['choiceLayer']
        self.exp_no = config['exp_no']

    def train(self, exp_no=0, useSLC=True):
        logger.info(f'Using device {self.device}')
        self.model.to(device=self.device)

        # self.model.to(device=self.device)
        # cudnn.benchmark = True

        optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)

        try:
            trainLoss = []
            trainReconLoss = []
            trainRLoss = []
            trainEdgeLoss = []
            trainVggLoss = []

            latencyThresholds = [100, 10, 9, 8, 7.5]
            binSize = self.epochs // len(latencyThresholds)
            binSize = 1

            for iter in range(self.epochs):
                trainLossSum = 0
                trainReconLossSum = 0
                trainRLossSum = 0
                trainEdgeLossSum = 0
                trainVggLossSum = 0

                idx = 0
                startTime = time.time()

                for LLow, LHigh, name in self.dataloader:
                    arch = [np.random.randint(self.choiceNum) for _ in range(self.choiceLayer)]

                    # calculate the latency
                    if useSLC is True:
                        archLatency = getLatency(arch, numChoice=self.choiceNum, numLayer=self.choiceLayer)
                        archLatencyThresholds = latencyThresholds[min(iter // binSize, len(latencyThresholds) - 1)]
                        while archLatency > archLatencyThresholds and np.random.randn(1) > 0.3:
                            arch = [np.random.randint(self.choiceNum) for _ in range(self.choiceLayer)]
                            archLatency = getLatency(arch)

                    LLow = LLow.to(self.device)
                    LHigh = LHigh.to(self.device)
                    RLow, ILow = self.model(LLow, arch=arch)
                    RHigh, IHigh = self.model(LHigh, arch=arch)

                    loss, lossComponents = self.lossFunction(RLow, RHigh, ILow, IHigh, LLow, LHigh)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if idx % self.printFrequency == 0:
                        logger.info(f"iter: {iter}_{idx}\t average_loss: {loss.item():.6f}\t")
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

                endTime = time.time()
                logger.info(f"iter: {iter}\t average_loss: {trainLoss[-1]:.6f}\t time:{endTime - startTime}")

                # save weight
                if iter % self.saveFrequency == 0:
                    if not os.path.exists(f'../weights/{self.exp_no}'):
                        os.mkdir(f'../weights/{self.exp_no}')
                    if not os.path.exists(f'../weights/{self.exp_no}/Supernet'):
                        os.mkdir(f'../weights/{self.exp_no}/Supernet')

                    torch.save(self.model.state_dict(),
                               f'../weights/{self.exp_no}/Supernet/checkpoint-{iter}.pth')

            torch.save(self.model.state_dict(),
                       f'../weights/{self.exp_no}/Supernet/checkpoint-latest.pth')

        except KeyboardInterrupt:
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)


if __name__ == "__main__":
    criterion = loss()
    model = HasNet()

    parser = Parser()
    args = parser.parse()
    args.config = "../Supernet/config.yaml"
    args.checkpoint = False
    if args.checkpoint:
        pretrain = torch.load('../weights/Supernet/checkpoint-latest.pth')
        model.load_state_dict(pretrain)
        logger.info('Model laded from DecomNet.pth')

    with open(args.config) as f:
        config = yaml.load(f)

    # trainPath = '/root/data/lyj/data/MIT5K/train'
    # vailPath = '/root/data/lyj/data/MIT5K/validation'
    # testPath = '/root/data/lyj/data/MIT5K/test'
    trainPath = '/Users/luoyongjia/Research/Data/MIT/train'
    vailPath = '/Users/luoyongjia/Research/Data/MIT/validation'
    testPath = '/Users/luoyongjia/Research/Data/MIT/train'
    trainListPath = buildDatasetListTxt(trainPath)
    vailListPath = buildDatasetListTxt(vailPath)
    testListPath = buildDatasetListTxt(testPath)
    logger.info("Building dataset...")
    trainData = loadDataset(trainPath, trainListPath, cropSize=config['length'], toRAM=True)
    vailData = loadDataset(vailPath, vailListPath, cropSize=config['length'], toRAM=True, training=False)
    testData = loadDataset(testPath, testListPath, cropSize=config['length'], toRAM=False, training=False)

    trainLoader = DataLoader(trainData, batch_size=config['batchSize'], shuffle=True)
    vailLoader = DataLoader(vailData, batch_size=1)
    testLoader = DataLoader(testData, batch_size=1)

    trainer = trainer(config, trainLoader, criterion, model)

    if args.mode == 'train':
        trainer.train()
