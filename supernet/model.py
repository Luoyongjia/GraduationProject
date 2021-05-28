import torch
import torch.nn as nn
from blocks.block import ChoiceBlock
from Utils.baseLayers import *


class mNet(nn.Module):
    def __init__(self, choiceNum=8, filters=32, activation='lrelu'):
        super(mNet, self).__init__()

        self.choiceNum = choiceNum
        self.convInput = Conv2D(3, filters)
        # R

        self.R_conv1 = Conv2D(filters, filters*2)
        self.R_down1 = ResidualDownSample(filters*2, filters*2)
        self.R_choice1 = self.buildChoiceBlock(filters*2, filters*4)
        self.R_down2 = ResidualDownSample(filters*4, filters*4)
        self.R_choice2 = self.buildChoiceBlock(filters*4, filters*8)
        self.R_down3 = ResidualDownSample(filters*8, filters*8)
        self.R_choice3 = self.buildChoiceBlock(filters*8, filters*16)
        self.R_up1 = convTranspose2D(filters*16, filters*8)
        self.concat1 = Concat()
        self.R_choice4 = self.buildChoiceBlock(filters*16, filters*8)
        self.R_up2 = convTranspose2D(filters*8, filters*4)
        self.concat2 = Concat()
        self.R_choice5 = self.buildChoiceBlock(filters*8, filters*4)
        self.R_up3 = convTranspose2D(filters*4, filters*2)
        self.concat3 = Concat()
        self.R_choice6 = self.buildChoiceBlock(filters*4, filters*4)
        self.R_choice7 = self.buildChoiceBlock(filters*4, filters*2)
        self.R_choice8 = self.buildChoiceBlock(filters*2, filters)
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

    def forward(self, x, arch):
        conv1 = self.convInput(x)

        # R
        R_conv1 = self.R_conv1(conv1)
        R_down1 = self.R_down1(R_conv1)
        R_chose1 = self.R_choice1[arch[0]](R_down1)
        R_down2 = self.R_down2(R_chose1)
        R_chose2 = self.R_choice2[arch[1]](R_down2)
        R_down3 = self.R_down3(R_chose2)
        R_chose3 = self.R_choice3[arch[2]](R_down3)
        R_up1 = self.R_up1(R_chose3)
        R_concat1 = self.concat1(R_chose2, R_up1)
        R_chose4 = self.choice4[arch[3]](R_concat1)
        R_up2 = self.R_up2(R_chose4)
        R_concat2 = self.concat2(R_chose1, R_up2)
        R_chose5 = self.choice5[arch[4]](R_concat2)
        R_up3 = self.R_up3(R_chose5)
        R_concat3 = self.concat2(R_conv1, R_up3)
        R_chose6 = self.choice6[arch[5]](R_concat3)
        R_chose7 = self.choice7[arch[6]](R_chose6)
        R_chose8 = self.choice8[arch[7]](R_chose7)
        R_out = self.R_sigmoid(R_chose8)

        # I
        I_conv1 = self.I_conv1(conv1)
        I_conv2 = self.I_conv2(I_conv1)
        I_conv3 = self.I_conv3(I_conv2)
        I_concat = self.I_concat(I_conv3, R_chose7)
        I_conv4 = self.I_ocnv4(I_concat)
        I_conv5 = self.I_conv5(I_conv4)
        I_conv6 = self.I_conv6(I_conv5)
        I_out = self.I_sigmoid(I_conv6)

        return R_out, I_out

    def buildChoiceBlock(self, inChannels, outChannels):
        choiceBlock = torch.nn.ModuleList()
        for choice in range(self.choiceNum):
            choiceBlock.append(ChoiceBlock(choice, inChannels, outChannels))

        return choiceBlock
