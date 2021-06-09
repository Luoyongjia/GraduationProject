from Model.blocks.choices import *


class ChoiceBlock(nn.Module):
    def __init__(self, arch, inChannels, outChannels):
        super(ChoiceBlock, self).__init__()
        if arch == 0:
            architecture = identity(inChannels, outChannels)
        elif arch == 1:
            architecture = Conv(inChannels, outChannels)
        elif arch == 2:
            architecture = doubleConv(inChannels, outChannels)
        elif arch == 3:
            architecture = residualBlock(inChannels, outChannels)
        elif arch == 4:
            architecture = dilConv(inChannels, outChannels, dilation=2)
        elif arch == 5:
            architecture = DenseBlock(inChannels, outChannels)
        elif arch == 6:
            architecture = channelAttention(inChannels, outChannels)
        elif arch == 7:
            architecture = spatialBlock(inChannels, outChannels, kernel_size=5)
        else:
            print("Not supported operation")

        self.architecture = architecture

    def forward(self, x):
        return self.architecture(x)
