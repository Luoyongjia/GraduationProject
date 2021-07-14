import torch
import torch.nn as nn
from torch import optim
from torch.optim import lr_scheduler
import time
import yaml
import sys
import cv2

import matplotlib as mpl
from matplotlib import pyplot as plt
mpl.use('Agg')

from skimage.metrics import structural_similarity, peak_signal_noise_ratio
from torchsummary import summary

from Model.loss import loss
from Model.model_retrain import HasNet
from Utils.Parser import Parser
from Utils.dataLoader import *
from Utils.utils import *

logger = log1('test', 'retrain')

class retrainer:
    def __init__(self, config, dataloader, criterion, model, dataloader_test):
        self.initialize(config)
        self.dataloader = dataloader
        self.dataloader_test = dataloader_test
        self.lossFunction = criterion
        self.model = model
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.model.to(device=self.device)
        if self.device == 'cuda':
            torch.backends.cudnn.benchmark = True
            self.lossFunction = criterion(device=self.device)

    def initialize(self, config):
        self.exp_no = config['exp_no']
        self.epochs = config['epochs']
        self.learningRate = config['learningRate']
        self.printFrequency = config['printFrequency']
        self.saveFrequency = config['saveFrequency']

    def train(self, exp_no=0):
        log(f'Using device {self.device}')
        logger.info(f'Using device {self.device}')

        self.model.to(device=self.device)
        optimizer = optim.Adam(self.model.parameters(), lr=self.learningRate)
        scheduler = lr_scheduler.ExponentialLR(optimizer, gamma=0.997)

        try:
            if not os.path.exists(f'./figure'):
                os.mkdir(f'./figure')
            if not os.path.exists(f'./figure/{self.exp_no}'):
                os.mkdir(f'./figure/{self.exp_no}')

            trainLoss = []
            trainReconLoss = []
            trainRLoss = []
            trainEdgeLoss = []
            trainVggLoss = []

            PSNRs =[]
            SSIMs = []

            for epoch in range(self.epochs):
                trainLossSum = 0
                trainReconLossSum = 0
                trainRLossSum = 0
                trainEdgeLossSum = 0
                trainVggLossSum = 0

                idx = 0

                startTime = time.time()

                for LLow, LHigh, name in self.dataloader:
                    LLow = LLow.to(self.device)
                    LHigh = LHigh.to(self.device)

                    RLow, ILow = self.model(LLow)
                    RHigh, IHigh = self.model(LHigh)

                    loss, lossComponents = self.lossFunction(RLow, RHigh, ILow, IHigh, LLow, LHigh)
                    optimizer.zero_grad()
                    loss.backward()
                    optimizer.step()

                    if idx % self.printFrequency == 0:
                        logger.info(f"iter: {epoch}_{idx}\t average_loss: {loss.item():.6f}\t")
                    idx += 1

                    # compute loss
                    trainLossSum += loss
                    trainReconLossSum += lossComponents['ReconLoss'].item()
                    trainRLossSum += lossComponents['RLoss'].item()
                    trainEdgeLossSum += lossComponents['EdgeLoss'].item()
                    trainVggLossSum += lossComponents['VggLoss'].item()

                # save loss
                trainLoss.append(loss.item())
                trainReconLoss.append(lossComponents['ReconLoss'].item())
                trainRLoss.append(lossComponents['RLoss'].item())
                trainEdgeLoss.append(lossComponents['EdgeLoss'].item())
                trainVggLoss.append(lossComponents['VggLoss'].item())

                PSNR, SSIM = self.test(epoch, plotDir=f'./figure/{self.exp_no}/',
                                       save_vis=(epoch % self.printFrequency == 0))
                PSNRs.append(PSNR)
                SSIMs.append(SSIM)

                # save weight
                if epoch % self.saveFrequency == 0:
                    if not os.path.exists(f'../weights/{self.exp_no}/retrain'):
                        os.mkdir(f'../weights/{self.exp_no}/retrain')

                    torch.save(self.model.state_dict(),
                               f'../weights/{self.exp_no}/retrain/checkpoint_{epoch}.pth')
                    log(f"Weights has been saved to '../weights/{self.exp_no}/retrain/checkpoint_{epoch}.pth")
                    logger.info(f"Weights has been saved to '../weights/{self.exp_no}/retrain/checkpoint_{epoch}.pth")
                scheduler.step()
                endTime = time.time()
                log(f'Time taken:{endTime - startTime: .3f} seconds\t lr={scheduler.get_lr()[0]:.6f}')
            # plot
            # loss
            t = np.arange(0, len(trainLoss), 1)
            print(t)
            print(trainLoss)
            plt.plot(t, trainLoss)
            plt.savefig(f'./figure/{self.exp_no}/trainloss.jpg')
            plt.show()
            # PSNR
            t = np.arange(0, len(PSNRs), 1)
            plt.plot(t, PSNRs)
            plt.savefig(f'./figure/{self.exp_no}/PSNRs.jpg')
            plt.show()
            # SSIM
            t = np.arange(0, len(SSIMs), 1)
            plt.plot(t, SSIMs)
            plt.savefig(f'./figure/{self.exp_no}/SSIMs.jpg')
            plt.show()

            # save convergence
            convergence = {'SSIMs': SSIMs, 'PSNRs': PSNRs,
                           'train_loss': trainLoss,
                           'ReconLoss': trainReconLoss, 'RLoss': trainRLoss, 'EdgeLoss': trainEdgeLoss,
                           'VggLoss': trainVggLoss}
            if not os.path.exists(f'../weights/{self.exp_no}/retrain/convergence'):
                os.mkdir(f'../weights/{self.exp_no}/retrain/convergence')

            torch.save(convergence, f'../weights/{self.exp_no}/retrain/convergence/checkpoint.pth')
            log(f"convergence has been saved to '../weights/{self.exp_no}/retrain/convergence/checkpoint.pth")
            logger.info(
                f"convergence has been saved to '../weights/{self.exp_no}/retrain/convergence/checkpoint.pth")

            # save last weight
            torch.save(self.model.state_dict(),
                       f'../weights/{self.exp_no}/retrain/checkpoint_latest.pth')
            log(f"Weights has been saved to '../weights/{self.exp_no}/retrain/checkpoint_lastest.pth")

        except KeyboardInterrupt:
            try:
                sys.exit(0)
            except SystemExit:
                os._exit(0)
    @torch.no_grad()
    def test(self, epoch=-1, max_iter=500, plotDir='./figure/', save_vis=False):
        self.model.eval()
        PSNRSum = 0
        SSIMSum = 0
        imageNum = 0
        imageSaveNum = 2

        for LLow, LHigh, name in self.dataloader_test:
            if not (imageNum < max_iter):
                break
            LLow_np = LLow.numpy()[0]
            LHigh_np = LHigh.numpy()[0]

            LLow = LLow.to(self.device)
            LHigh = LHigh.to(self.device)

            RLow, ILow = self.model(LLow)
            RHigh, IHigh = self.model(LHigh)

            RLow_np = RLow.detach().cpu().numpy()[0]
            RHigh_np = RHigh.detach().cpu().numpy()[0]
            ILow_np = ILow.detach().cpu().numpy()[0]
            IHigh_np = IHigh.detach().cpu().numpy()[0]

            if save_vis and imageNum < imageSaveNum:
                sample_imgs = np.concatenate((RLow_np, ILow_np, LLow_np,
                                              RHigh_np, IHigh_np, LHigh_np), axis=0)

                filepath = os.path.join(plotDir, f'{name[0]}_epoch_{epoch}.png')
                split_point = [0, 3, 6, 9, 12, 15, 18]
                img_dim = ILow_np.shape[1:]
                sample(sample_imgs, split=split_point, figure_size=(2, 3),
                       img_dim=img_dim, path=filepath, num=epoch)

            # compute PSNR, SSIM
            PSNRSum += peak_signal_noise_ratio(RLow_np, LHigh_np)
            SSIMSum += structural_similarity(cv2.cvtColor(np.transpose(RLow_np, [1, 2, 0]), cv2.COLOR_BGR2GRAY),
                                             cv2.cvtColor(np.transpose(LHigh_np, [1, 2, 0]), cv2.COLOR_BGR2GRAY))
            imageNum += 1

        print(f'PSNR: {PSNRSum / imageNum}, SSIM: {SSIMSum / imageNum}')

        return PSNRSum / imageNum, SSIMSum / imageNum

    def get_illumination(self, epoch=-1, plot_dir='./figure/'):
        for LLow, LHigh, name in self.dataloader_test:
            ILow_gt = LLow / torch.max(LHigh, torch.ones_like(LHigh))
            LLow_np = LLow.numpy()[0]
            LHigh_np = LHigh.numpy()[0]
            ILow_gt = ILow_gt.numpy()[0]

            filepath = os.path.join(plot_dir, f'{name[0]}_epoch_{epoch}.png')
            sampleImages = np.concatenate((LHigh_np, ILow_gt, LLow_np), axis=0)
            splitPoint = [0, 3, 6, 9]
            imageDim = LLow_np.shape[1:]
            sample(sampleImages, split=splitPoint, figure_size=(1, 3),
                   img_dim=imageDim, path=filepath)


if __name__ == "__main__":

    parser = Parser()
    args = parser.parse()
    args.config = "../Retrain/config.yaml"

    with open(args.config) as f:
        config = yaml.load(f)

    # load searched architectures
    archCheck = torch.load(config['archDir'])
    arch = list(archCheck['topKDict'][10][0])
    model = HasNet(arch=arch)
    criterion = loss()

    # check point load
    if config['checkPoint']:
        pretrain = torch.load(config['checkPointDir'])
        model.load_state_dict(pretrain)
        log(f"Model loaded from {config['checkPointDir']}")
        logger.info(f"Model loaded from {config['checkPointDir']}")

    # load data
    trainPath = '/Users/luoyongjia/Research/Data/MIT/train'
    testPath = '/Users/luoyongjia/Research/Data/MIT/train'

    trainListPath = buildDatasetListTxt(trainPath)
    testListPath = buildDatasetListTxt(testPath)

    log("Building dataset...")
    logger.info("Building dataset...")
    trainData = loadDataset(trainPath, trainListPath, cropSize=config['length'], toRAM=False)
    testData = loadDataset(testPath, testListPath, cropSize=config['length'], toRAM=False, training=False)

    trainLoader = DataLoader(trainData, batch_size=config['batchSize'], shuffle=True)
    testLoader = DataLoader(testData, batch_size=1)

    trainer = retrainer(config, trainLoader, criterion, model, dataloader_test=testLoader)

    # train
    trainer.train(exp_no=config['exp_no'])
    checkPoint = torch.load(f"../weights/{config['exp_no']}/retrain/convergence/checkpoint.pth")
    log(f"Max PSNR: {np.max(checkPoint['PSNRs'])} \t Max SSIM: {np.max(checkPoint['SSIMs'])}")
    for key in checkPoint.keys():
        if len(checkPoint[key]) != 0:
            plt.xlabel("epoches")
            plt.ylabel(key)
            plt.plot(range(len(checkPoint[key])), checkPoint[key], label=key)
            plt.savefig(f"../weights/{config['exp_no']}/retrain/convergence/{key}.png")
            plt.clf()



