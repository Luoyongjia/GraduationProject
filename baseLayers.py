import torch
import torch.nn as nn
import torch.nn.functional as F


class Conv2D(nn.Module):
    def __init__(self, inChannels, outChannels, activation='lRelu', stride=1):
        super(Conv2D, self).__init__()
        self.activationLayer = nn.LeakyReLU(inplace=True)
        if activation == 'relu':
            self.activationLayer = nn.ReLU(inplace=True)
        self.convRelu = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=stride, padding=1),
            self.activationLayer,
        )

    def forward(self, x):
        return self.convRelu(x)


class MaxPooling2D(nn.Module):
    def __init__(self, kernelSize=2, stride=2):
        super(MaxPooling2D, self).__init__()
        self.maxpool = nn.MaxPool2d(kernel_size=kernelSize, stride=stride)

    def forward(self, x):
        return self.maxpool(x)


class deConv2D(nn.Module):
    def __init__(self, inChannels, outChannels, activation='lRelu'):
        super(deConv2D, self).__init__()
        self.activation = nn.LeakyReLU(inplace=True)
        if activation == 'Relu':
            self.activation = nn.ReLU(inplace=True)
        self.deConvRelu = nn.Sequential(
            nn.ConvTranspose2d(inChannels, outChannels, kernel_size=2, stride=2, padding=0),
            self.activation,
        )

    def forward(self, x):
        return self.deConvRelu(x)


class Concat(nn.Module):
    def forward(self, x, y):
        _, _, xh, xw = x.size()
        _, _, yh, yw = y.size()
        diffY = xh - yh
        diffX = xw - yw
        y = F.pad(y, [diffX//2, diffX - diffX//2, diffY//2, diffY - diffY//2])
        return torch.cat((x, y), dim=1)
