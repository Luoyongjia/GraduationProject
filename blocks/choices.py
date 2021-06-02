import torch
import torch.nn as nn

from Utils.baseLayers import *
from Utils.utils import GCD


class identity(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(identity, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0)
        self.activation = nn.PReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        return x


class Conv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(Conv, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1)
        self.activation = nn.ReLU()

    def forward(self, x):
        x = self.conv1(x)
        x = self.activation(x)

        return x


class doubleConv(nn.Module):
    def __init__(self, inChannels, outChannels):
        super(doubleConv, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU())
        self.conv2 = nn.Sequential(
            nn.Conv2d(outChannels, outChannels, kernel_size=3, stride=1, padding=1),
            nn.ReLU())

    def forward(self, x):
        x = self.conv1(x)
        x = self.conv2(x)

        return x


class residualBlock(nn.Module):
    def __init__(self, inChannel, outChannel):
        super(residualBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1, groups=inChannel),
            nn.Conv2d(inChannel, inChannel, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=1, groups=GCD(inChannel, outChannel)),
            nn.Conv2d(outChannel, outChannel, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )
        self.channelFusion = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

    def forward(self, x):
        out = self.layer1(x)
        out = self.layer2(out)

        fusion = self.channelFusion(x)

        return fusion + out


class dilConv(nn.Module):
    def __init__(self, inChannel, outChannel, dilation=1):
        super(dilConv, self).__init__()
        self.left = nn.Sequential(
            nn.ReLU(inplace=False),
            nn.Conv2d(inChannel, outChannel, kernel_size=3, stride=1, padding=2, dilation=dilation),
            nn.Conv2d(outChannel, outChannel, kernel_size=1, padding=0, bias=False),
            nn.ReLU()
        )
        self.right = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)

        return left + right


class DenseBlock(nn.Module):
    """Residual Dens Block"""
    def __init__(self, inChannel, outChannels):
        super(DenseBlock, self).__init__()
        self.layer1 = nn.Sequential(
            nn.Conv2d(inChannel, inChannel, kernel_size=3, stride=1, padding=1, groups=inChannel),
            nn.Conv2d(inChannel, inChannel, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )
        self.layer2 = nn.Sequential(
            nn.Conv2d(inChannel*2, inChannel, kernel_size=3, stride=1, padding=1, groups=inChannel),
            nn.Conv2d(inChannel, inChannel, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )
        self.layer3 = nn.Sequential(
            nn.Conv2d(inChannel * 3, outChannels, kernel_size=3, stride=1, padding=1, groups=GCD(inChannel*3, outChannels)),
            nn.Conv2d(outChannels, outChannels, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

        self.lReLU = nn.PReLU()
        self.channelFusion = nn.Sequential(
            nn.Conv2d(inChannel, outChannels, kernel_size=1, stride=1, padding=0),
            nn.PReLU(),
        )

    def forward(self, x):
        x1 = self.layer1(x)
        x2 = self.layer2(torch.cat((x, x1), 1))
        x3 = self.layer3(torch.cat((x, x1, x2), 1))

        x4 = self.channelFusion(x)

        return x3*0.333 + x4


class channelAttention(nn.Module):
    """
    ECA blocks, supply channel attention

    Adjust the channel number at first, the calculate channel is out channels
    """
    def __init__(self, inChannels, outChannels, kernel_size=3, reduction=64,):
        super(channelAttention, self).__init__()
        self.conv1 = nn.Conv2d(inChannels, outChannels, kernel_size=3, stride=1, bias=False)
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        # channel based padding size
        self.conv2 = nn.Conv1d(1, 1, kernel_size=kernel_size, padding=(kernel_size - 1) // 2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        x = self.conv1(x)
        y = self.avg_pool(x)
        y = self.conv2(y.squeeze(-1).transpose(-1, -2)).transpose(-1, -2).unsqueeze(-1)
        y = self.sigmoid(y)

        return x*y.expand_as(x)


class spatialBlock(nn.Module):
    """
    2d attention
    """
    def __init__(self, inChannel, outChannel, kernel_size=5):
        super(spatialBlock, self).__init__()

        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, stride=1, padding=2)
        self.sigmoid = nn.Sigmoid()

        self.channelFusion = nn.Sequential(
            nn.Conv2d(inChannel, outChannel, kernel_size=1, stride=1, padding=0),
            nn.PReLU()
        )

    def forward(self, x):
        x = self.channelFusion(x)
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.conv(out)
        scale = self.sigmoid(out)
        print(out.size())
        x_out = x * scale

        return x_out