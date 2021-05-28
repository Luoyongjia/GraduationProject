import os
import shutil
import random
import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from Utils.utils import *

import matplotlib.pyplot as plt


class CustomDataset(Dataset):
    def __init__(self, dataPath):
        super().__init__()
        self.dataPath = dataPath
        self.imgPath = [os.path.join(dataPath, f)
                        for f in os.listdir(dataPath)
                        if any(fileType in f.lower()
                               for fileType in ['jpeg', 'png', 'jpg', 'bmp'])]
        self.name = [f.split(".")[0]
                     for f in os.listdir(dataPath)
                     if any(fileType in f.lower()
                            for fileType in ['jpeg', 'png', 'jpg', 'bmp'])]

    def __len__(self):
        return len(self.imgPath)

    def __getiem__(self, idx):
        dataFiles = self.imgPath[idx]
        img = Image.open(dataFiles).covert('RGB')
        img = np.array(img, np.float32).transpose((2, 0, 1)) / 255.
        return img, self.name[idx]


class loadDataset(Dataset):
    def __init__(self, root, listPath, cropSize=256, toRAM=False, training=True):
        super(loadDataset, self).__init__()
        self.root = root            # 路径
        self.listPath = listPath    # 数据名称list
        self.cropSize = cropSize
        self.toRAM = toRAM
        self.training = training
        with open(listPath) as f:
            self.pairs = f.readlines()
        self.files = []
        self.firstLine = True

        for pair in self.pairs:
            if self.firstLine:
                self.firstLine = False
                continue
            lowPath, highPath = pair.split(",")
            highPath = highPath[:-1]
            name = lowPath.split("\\")[-1][:-4]
            lowFile = os.path.join(self.root, lowPath)
            highFile = os.path.join(self.root, highPath)
            self.files.append({
                "lr": lowFile,
                "hr": highFile,
                "name": name
            })
        self.data = []
        if self.toRAM:
            for i, fileInfo in enumerate(self.files):
                name = fileInfo["name"]
                lowImg = Image.open(fileInfo["lr"])
                lowImg = lowImg.resize((400, 600))
                highImg = Image.open(fileInfo["hr"])
                highImg = highImg.resize((400, 600))

                self.data.append({
                    "lr": lowImg,
                    "hr": highImg,
                    "name": name
                })
            log("Finish loading all images to RAM...")

    def __len__(self):
        return len(self.files)

    def __getitem__(self, idx):
        dataFiles = self.files[idx]

        '''load the data'''
        if self.toRAM:
            name = self.data[idx]["name"]
            lowImg = self.data[idx]["lr"]
            highImg = self.data[idx]["hr"]
        else:
            name = dataFiles["name"]
            lowImg = Image.open(dataFiles["lr"])
            highImg = Image.open(dataFiles["hr"])
            lowImg = lowImg.resize((400, 600))
            highImg = highImg.resize((400, 600))

        '''random crop the inputs'''
        if self.cropSize > 0:

            # select a random start point
            hOffset = random.randint(0, lowImg.size[1] - self.cropSize)
            wOffset = random.randint(0, lowImg.size[0] - self.cropSize)

            # crop the image and the label
            cropBox = (wOffset, hOffset, wOffset+self.cropSize, hOffset+self.cropSize)
            lowCrop = lowImg
            highCrop = highImg
            if self.training is True:
                lowCrop = lowImg.crop(cropBox)
                highCrop = highImg.crop(cropBox)
                randMode = np.random.randint(0, 7)
                lowCrop = dataAugmentation(lowCrop, randMode)
                highCrop = dataAugmentation(highCrop, randMode)

            '''covert PIL Image to numpy array'''
            lowCrop = np.asarray(lowCrop, np.float32).transpose((2, 0, 1)) / 255.
            highCrop = np.asarray(highCrop, np.float32).transpose((2, 0, 1)) / 255.
            return lowCrop, highCrop, name


def buildDatasetListTxt(dstDir):
    """
    Creating the list of images，saving in pairList.csv
    :param dstDir: the path of images and saving the csv file
    :return: listPath
    """
    log(f"Building dataset list text at {dstDir}")
    lDir = os.path.join(dstDir, 'low')
    hDir = os.path.join(dstDir, 'high')
    imgLPath = [os.path.join('low', name) for name in os.listdir(lDir)]
    imgRPath = [os.path.join('high', name) for name in os.listdir(hDir)]
    listPath = os.path.join(dstDir, 'pairList.csv')
    with open(listPath, 'w') as f:
        for lPath, hPath in zip(imgLPath, imgRPath):
            f.write(f"{lPath},{hPath}\n")
        log(f"Finish... There are {len(imgLPath)}pairs...")
        return listPath



if __name__ == '__main__':
    trainPath = '/Users/luoyongjia/Research/Data/MIT/train'
    vailPath = '/Users/luoyongjia/Research/Data/MIT/validation'
    testPath = '/Users/luoyongjia/Research/Data/MIT/test'
    trainListPath = buildDatasetListTxt(trainPath)
    vailListPath = buildDatasetListTxt(vailPath)
    testListPath = buildDatasetListTxt(testPath)
    log("Building dataset...")
    BatchSize = 2
    trainData = loadDataset(trainPath, trainListPath, cropSize=128, toRAM=True)
    vailData = loadDataset(vailPath, vailListPath, cropSize=128, toRAM=False)
    testData = loadDataset(testPath, testListPath, cropSize=128, toRAM=False)

    trainLoader = DataLoader(trainData, batch_size=BatchSize)
    vailLoader = DataLoader(vailData, batch_size=BatchSize)
    testLoader = DataLoader(testData, batch_size=BatchSize)

    # plt.ion()
    # for i, data in enumerate(vailLoader):
    #     imgs, name = data
    #     img = imgs[0].numpy()
    #     sample(imgs[0], figure_size=(1, 1), img_dim=128)
