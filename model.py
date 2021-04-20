import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from baseLayers import *

class mNet(nn.Module):
    def __init__(self, filters=32, activation='lrelu'):
        self.convInput = Conv2D(3, filters)
        # R
        self.convR1 = Conv2D(filters, filters*2)
        self.maxpoolR1 = MaxPooling2D()
        self.convR2 = Conv2D(filters*2, filters*4)
        self.maxpoolR2 = MaxPooling2D()
        self.deConvR1 = deConv2D(filters*4, filters*2)
        self.concatR1 = Concat()
        self.convR3 = Conv2D(filters*4, filters*2)
        self.deConvR2 = deConv2D(filters*2, filters)
        self.concatR2 = Concat()
        self.convR4 = Conv2D(filters*2, filters)
        self.convR5 = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self.ROut = nn.Sigmoid()

        # L
        self.convI1 = Conv2D(filters, filters)
        self.concatI1 = Concat()
        self.convI2 = nn.Conv2d(filters*2, 1, kernel_size=3, padding=1)
        self.Iout = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.convInput(x)

        # R
        maxpoolR1 = self.maxpoolR1(conv1)
        convR2 = self.convR2(maxpoolR1)
        maxpoolR2 = self.maxpoolR2(convR2)
        convR3 = self.convR2(maxpoolR2)
        deConvR1 = self.deConvR1(convR3)
        concatR1 = self.concatR1(deConvR1, convR2)
        convR4 = self.convR3(concatR1)
        deConvR2 = self.deConvR2(convR4)
        concatR2 = self.concatR2(deConvR2, conv1)
        convR5 = self.convR4(concatR2)
        convR6 = self.convR5(convR5)
        ROut = self.ROut(convR6)

        #I
        convI1 = self.convI1(conv1)
        concatI1 = self.concatI1(convI1, convR5)
        convI2 = self.convI2(concatI1)
        IOut = self.Iout(convI2)
