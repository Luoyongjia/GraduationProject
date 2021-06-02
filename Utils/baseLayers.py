import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


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


class convTranspose2D(nn.Module):
    def __init__(self, inChannels, outChannels, activation='lRelu'):
        super(convTranspose2D, self).__init__()
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


class BlurPool(nn.Module):
    """
    https://github.com/adobe/antialiased-cnns
    proposed pooling method
    ./antialiased_cnns/blurpool.py
    """
    def __init__(self, channels, pad_type='reflect', filt_size=4, stride=2, pad_off=0):
        super(BlurPool, self).__init__()
        self.filt_size = filt_size
        self.pad_off = pad_off
        self.pad_sizes = [int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2)), int(1.*(filt_size-1)/2), int(np.ceil(1.*(filt_size-1)/2))]
        self.pad_sizes = [pad_size+pad_off for pad_size in self.pad_sizes]
        self.stride = stride
        self.off = int((self.stride-1)/2.)
        self.channels = channels

        # define the blur kernel
        if self.filt_size == 1:
            a = np.array([1., ])
        elif self.filt_size == 2:
            a = np.array([1., 1.])
        elif self.filt_size == 3:
            a = np.array([1., 2., 1.])
        elif self.filt_size == 4:
            a = np.array([1., 3., 3., 1.])
        elif self.filt_size == 5:
            a = np.array([1., 4., 6., 4., 1.])
        elif self.filt_size == 6:
            a = np.array([1., 5., 10., 10., 5., 1.])
        elif self.filt_size == 7:
            a = np.array([1., 6., 15., 20., 15., 6., 1.])

        filt = torch.Tensor(a[:, None]*a[None, :])
        filt = filt/torch.sum(filt)
        self.register_buffer('filt', filt[None, None, :, :].repeat((self.channels, 1, 1, 1)))

        self.pad = self.get_pad_layer(pad_type)(self.pad_sizes)

    def forward(self, inp):
        if self.filt_size == 1:
            if self.pad_off == 0:
                return inp[:, :, ::self.stride, ::self.stride]
            else:
                return self.pad(inp)[:, :, ::self.stride, ::self.stride]
        else:
            return F.conv2d(self.pad(inp), self.filt, stride=self.stride, groups=inp.shape[1])

    def get_pad_layer(self, pad_type):
        if pad_type in ['refl', 'reflect']:
            PadLayer = nn.ReflectionPad2d
        elif pad_type in ['repl', 'replicate']:
            PadLayer = nn.ReplicationPad2d
        elif pad_type == 'zero':
            PadLayer = nn.ZeroPad2d
        else:
            print('Pad type [%s] not recognized' % pad_type)

        return PadLayer


class ResidualDownSample(nn.Module):
    def  __init__(self, inChannels, outChannels, bias=False):
        super(ResidualDownSample, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inChannels, inChannels, kernel_size=3, stride=1, padding=1, bias=bias),
            nn.ReLU(),
            BlurPool(channels=inChannels, filt_size=3, stride=2),
            nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0, bias=bias)
        )
        self.right = nn.Sequential(
            BlurPool(channels=inChannels, filt_size=3, stride=2),
            nn.Conv2d(inChannels, outChannels, kernel_size=1, stride=1, padding=0, bias=bias)
        )

    def forward(self, x):
        left = self.left(x)
        right = self.right(x)

        return left + right
