import torch
import torch.nn as nn
from Model.blocks.block import ChoiceBlock
from Utils.baseLayers import *


class HasNet(nn.Module):
    def __init__(self, arch, filters=32, activation='lrelu'):
        super(HasNet, self).__init__()
        assert(len(arch) == 8)

        self.convInput = Conv2D(3, filters)
        # R

        self.R_conv1 = Conv2D(filters, filters*2)
        self.R_down1 = ResidualDownSample(filters*2, filters*2)
        self.R_choice1 = self.buildChoiceBlock(arch[0], filters*2, filters*4)
        self.R_down2 = ResidualDownSample(filters*4, filters*4)
        self.R_choice2 = self.buildChoiceBlock(arch[1], filters*4, filters*8)
        self.R_down3 = ResidualDownSample(filters*8, filters*8)
        self.R_choice3 = self.buildChoiceBlock(arch[2], filters*8, filters*16)
        self.R_up1 = convTranspose2D(filters*16, filters*8)
        self.concat1 = Concat()
        self.R_choice4 = self.buildChoiceBlock(arch[3], filters*16, filters*8)
        self.R_up2 = convTranspose2D(filters*8, filters*4)
        self.concat2 = Concat()
        self.R_choice5 = self.buildChoiceBlock(arch[4], filters*8, filters*4)
        self.R_up3 = convTranspose2D(filters*4, filters*2)
        self.concat3 = Concat()
        self.R_choice6 = self.buildChoiceBlock(arch[5], filters*4, filters*2)
        self.R_choice7 = self.buildChoiceBlock(arch[6], filters*2, filters*1)
        self.R_choice8 = self.buildChoiceBlock(arch[7], filters, 3)
        self.R_sigmoid = nn.Sigmoid()

        # I
        self.I_conv1 = Conv2D(filters, filters)
        self.I_conv2 = Conv2D(filters, filters)
        self.I_conv3 = Conv2D(filters, filters)
        self.I_concat = Concat()
        self.I_conv4 = Conv2D(filters*2, filters)
        self.I_conv5 = Conv2D(filters, filters)
        self.I_conv6 = nn.Conv2d(filters, 3, kernel_size=3, padding=1)
        self.I_sigmoid = nn.Sigmoid()

    def forward(self, x):
        conv1 = self.convInput(x)

        # R
        R_conv1 = self.R_conv1(conv1)
        R_down1 = self.R_down1(R_conv1)
        R_chose1 = self.R_choice1(R_down1)
        R_down2 = self.R_down2(R_chose1)
        R_chose2 = self.R_choice2(R_down2)
        R_down3 = self.R_down3(R_chose2)
        R_chose3 = self.R_choice3(R_down3)
        R_up1 = self.R_up1(R_chose3)
        R_concat1 = self.concat1(R_chose2, R_up1)
        R_chose4 = self.R_choice4(R_concat1)
        R_up2 = self.R_up2(R_chose4)
        R_concat2 = self.concat2(R_chose1, R_up2)
        R_chose5 = self.R_choice5(R_concat2)
        R_up3 = self.R_up3(R_chose5)
        R_concat3 = self.concat2(R_conv1, R_up3)
        R_chose6 = self.R_choice6(R_concat3)
        R_chose7 = self.R_choice7(R_chose6)
        R_chose8 = self.R_choice8(R_chose7)
        R_out = self.R_sigmoid(R_chose8)

        # I
        I_conv1 = self.I_conv1(conv1)
        I_conv2 = self.I_conv2(I_conv1)
        I_conv3 = self.I_conv3(I_conv2)
        I_concat = self.I_concat(I_conv3, R_chose7)
        I_conv4 = self.I_conv4(I_concat)
        I_conv5 = self.I_conv5(I_conv4)
        I_conv6 = self.I_conv6(I_conv5)
        I_out = self.I_sigmoid(I_conv6)

        return R_out, I_out

    def buildChoiceBlock(self, op, inChannels, outChannels):
        choiceBlock = ChoiceBlock(op, inChannels, outChannels)

        return choiceBlock
