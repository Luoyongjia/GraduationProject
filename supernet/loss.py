import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Utils.dataLoader import *
from Utils.networks import Vgg16


SOBEL = np.array([[-1, -2, -1],
                  [0, 0, 0],
                  [1, 2, 1]])
ROBERT = np.array([[0, 0],
                   [-1, 1]])

SOBEL = torch.Tensor(SOBEL)
ROBERT = torch.Tensor(ROBERT)

def featureMapHook(*args, path=None):
    featureMaps = []
    for feature in args:
        featureMaps.append(feature)
    featureAll = torch.cat(featureMaps, dim=1)
    fMap = featureAll.detach().cpu().numpy()[0]
    fMap = np.array(fMap)
    fShape = fMap.shape
    num = fShape[0]
    shape = fShape[1:]
    sample(fMap, figure_size=(2, num//2), img_dim=shape, path=path)

# 提取一阶导数算子滤波图
def gradient(maps, direction, device='cpu', kernel='SOBEL', abs='True'):
    channels = maps.size()[1]
    if kernel == 'ROBERT':
        smoothKernelx = ROBERT.expand(channels, channels, 2, 2)
        maps = F.pad(maps, [0, 0, 1, 1])
    elif kernel == 'SOBEL':
        smoothKernelx = SOBEL.expand(channels, channels, 3, 3)
        maps = F.pad(maps, [1, 1, 1, 1])
    smoothKernely = smoothKernelx.permute(0, 1, 3, 2)
    if direction == "x":
        kernel = smoothKernelx
    elif direction == "y":
        kernel = smoothKernely
    kernel = kernel.to(device=device)
    if abs:
        gradientOrig = torch.abs(F.conv2d(maps, weight=kernel, padding=0))
    else:
        gradientOrig = F.conv2d(maps, weight=kernel, padding=0)
    gradMin = torch.min(gradientOrig)
    gradMax = torch.max(gradientOrig)
    gradNorm = torch.div((gradientOrig - gradMin), (gradMax - gradMin + 0.0001))
    return gradNorm


class loss(nn.Module):
    def __init__(self):
        super(loss, self).__init__()

    def reconstructionLoss(self, RLow, ILow3, LLow):
        reconLossLow = torch.mean(torch.abs(RLow * ILow3 - LLow))
        return reconLossLow

    def RLoss(self, rLow, rHigh):
        # torch.norm(rLow - rHigh, p=2)
        return torch.mean(torch.abs(rLow - rHigh))

    def edgePreservingLoss(self, ILow, IHigh, hook=-1):
        lowGradientx = gradient(ILow, "x")
        highGradientx = gradient(IHigh, "x")
        MGradientx = lowGradientx + highGradientx
        xLoss = MGradientx * torch.exp(-10 * MGradientx)

        lowGradienty = gradient(ILow, "y")
        highGradienty = gradient(IHigh, "y")
        MGradienty = lowGradienty + highGradienty
        yLoss = MGradienty * torch.exp(-10 * MGradienty)

        mutualLoss = torch.mean(xLoss + yLoss)

        return mutualLoss

    def VggLoss(self, RLow, LHigh):
        instancenorm = nn.InstanceNorm2d(512, affine=False)
        # process input images, convert RGB, BGR
        (RLow_r, RLow_g, RLow_b) = torch.chunk(RLow, 3, dim=1)
        (LHigh_r, LHigh_g, LHigh_b) = torch.chunk(RLow, 3, dim=1)
        RLow = torch.cat((RLow_b, RLow_g, RLow_r), dim=1)
        LHigh = torch.cat((LHigh, LHigh, LHigh), dim=1)

        # load vgg16
        vgg = Vgg16()
        vgg.cuda()
        vgg.load_state_dict(torch.load('../Utils/vgg16.weight'))
        vgg.eval()
        for param in vgg.parameters():
            param.requires_grad = False
        RLowFeature = vgg(RLow)
        LHighFeature = vgg(LHigh)

        vggLoss = torch.mean(instancenorm(RLowFeature) - instancenorm(LHighFeature) ** 2)

        return vggLoss

    def forward(self, RLow, RHigh, ILow, IHigh, LLow, LHigh, hook=-1):
        ILow3 = torch.cat([ILow, ILow, ILow], dim=1)

        recLoss = self.reconstructionError(RLow, ILow3, LLow)
        rLoss = self.RLoss(RLow, LHigh)
        edgeLoss = self.edgePreservingLoss(ILow, IHigh, hook=hook)
        vggLoss = self.VggLoss(RLow, LHigh)

        Loss = 0.3*recLoss + 0.5 * rLoss + 0.2 * edgeLoss + 0.002 * vggLoss
        return Loss
