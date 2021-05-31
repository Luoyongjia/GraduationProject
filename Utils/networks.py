import torch
import torch.nn as nn
import torch.nn.functional as F


class Vgg16(nn.Module):
    def __init__(self):
        super(Vgg16, self).__init__()
        self.conv1_1 = nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1)
        self.conv1_2 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)

        self.conv2_1 = nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1)
        self.conv2_2 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)

        self.conv3_1 = nn.Conv2d(128, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_2 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)
        self.conv3_3 = nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1)

        self.conv4_1 = nn.Conv2d(256, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv4_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

        self.conv5_1 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_2 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)
        self.conv5_3 = nn.Conv2d(512, 512, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.conv1_1(x)
        x = F.relu(x, inplace=True)
        x = self.conv1_2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv2_1(x)
        x = F.relu(x, inplace=True)
        x = self.conv2_2(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv3_1(x)
        x = F.relu(x, inplace=True)
        x = self.conv3_2(x)
        x = F.relu(x, inplace=True)
        x = self.conv3_3(x)
        x = F.relu(x, inplace=True)
        x = F.max_pool2d(x, kernel_size=2, stride=2)

        x = self.conv4_1(x)
        x = F.relu(x, inplace=True)
        x = self.conv4_2(x)
        x = F.relu(x, inplace=True)
        x = self.conv4_3(x)
        x = F.relu(x, inplace=True)

        x = self.conv5_1(x)
        out = F.relu(x, inplace=True)
        x = self.conv5_2(out)
        x = F.relu(x)
        x = self.conv5_3(x)
        x = F.relu(x, inplace=True)
        relu5_3 = x
        return out
