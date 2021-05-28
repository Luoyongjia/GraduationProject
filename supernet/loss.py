import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from Utils.dataLoader import *


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
        super().__init__()

    def reflectanceSimilarity(self, rLow, rHigh):
        # torch.norm(rLow - rHigh, p=2)
        return torch.mean(torch.abs(rLow - rHigh))

    def illuminationSmoothness(self, I, L, name='low', hook=-1):
        LGray = 0.299 * L[:, 0, :, :] + 0.587 * L[:, 1, :, :] + 0.114 * L[:, 2, :, :]
        LGray = LGray.unsqueeze(dim=1)

        IGradientx = gradient(I, "x")
        LGradientx = gradient(LGray, "x")
        epsilon = 0.01 * torch.ones_like(LGradientx)
        denominatorx = torch.max(LGradientx, epsilon)
        xLoss = torch.abs(torch.div(IGradientx, denominatorx))

        IGradienty = gradient(I, "y")
        LGradienty = gradient(LGray, "y")
        epsilon = 0.01 * torch.ones_like(LGradienty)
        denominatory = torch.max(LGradienty, epsilon)
        yLoss = torch.abs(torch.div(IGradienty, torch.max(LGradienty, denominatory)))

        mutLoss = torch.mean(xLoss + yLoss)

        if hook > -1:
            featureMapHook(I, LGray, epsilon, IGradientx + IGradienty, denominatorx + denominatory,
                           xLoss + yLoss, path=f'./images/samples-features/iluxSmooth{name}epoch{hook}.png')
        return mutLoss

    def mutalConsistency(self, ILow, IHigh, hook=-1):
        lowGradientx = gradient(ILow, "x")
        highGradientx = gradient(IHigh, "x")
        MGradientx = lowGradientx + highGradientx
        xLoss = MGradientx * torch.exp(-10 * MGradientx)

        lowGradienty = gradient(ILow, "y")
        highGradienty = gradient(IHigh, "y")
        MGradienty = lowGradientx + highGradienty
        yLoss = MGradienty * torch.exp(-10 * MGradienty)

        mutualLoss = torch.mean(xLoss + yLoss)

        # if hook > -1:
        #     featureMapHook(ILow, IHigh, lowGradientx+lowGradienty, highGradientx + highGradienty,
        #             MGradientx + MGradienty, xLoss + yLoss,
        #                    path=f'./images/samples-features/mutual_consist_epoch{hook}.png')
        return mutualLoss

    def reconstructionError(self, RLow, RHigh, ILow3, IHigh3, LLow, LHigh):
        reconLossLow = torch.mean(torch.abs(RLow * ILow3 - LLow))
        reconLossHigh = torch.mean(torch.abs(RHigh * IHigh3 - LHigh))
        return reconLossLow + reconLossHigh

    def forward(self, RLow, RHigh, ILow, IHigh, LLow, LHigh, hook=-1):
        ILow3 = torch.cat([ILow, ILow, ILow], dim=1)
        IHigh3 = torch.cat([IHigh, IHigh, IHigh], dim=1)

        recLoss = self.reconstructionError(RLow, RHigh, ILow3, IHigh3, LLow, LHigh)
        rsLoss = self.reflectanceSimilarity(RLow, RHigh)
        isLoss = self.illuminationSmoothness(IHigh, LLow, hook=hook) + \
                 self.illuminationSmoothness(IHigh, LHigh, name='high', hook=hook)
        mcLoss = self.mutalConsistency(ILow, IHigh, hook=hook)

        Loss = recLoss + 0.01 * rsLoss + 0.08 * isLoss + 0.1 * mcLoss
        return Loss
