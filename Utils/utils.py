import os
import time
import numpy as np
import cv2


def GCD(a, b):
    """
    calculate the greatest common divisor
    """
    while b:
        a, b = b, a % b
    return a


def log(string):
    """
    add the time information before the log
    :param string:
    :return:
    """
    print(time.strftime('%H:%M:%S'), ">>", string)


def dataAugmentation(image, mode):
    """
    some operation of images
    :param image: the image needed some operations
    :param mode: the kinds of operations
    :return: the operated image
    """
    if mode == 0:
        # original
        return image
    elif mode == 1:
        # flip up and down
        return np.flipud(image)
    elif mode == 2:
        # rotate 90 degrees counterclockwise
        return np.rot90(image)
    elif mode == 3:
        image = np.rot90(image)
        return np.flipud(image)
    elif mode == 4:
        return np.rot90(image, k=2)
    elif mode == 5:
        image = np.rot90(image, k=2)
        return np.flipud(image)
    elif mode == 6:
        return np.rot90(image, k=3)
    elif mode == 7:
        image = np.rot90(image, k=3)
        return np.flipud(image)


def sample(imgs, split=None, figure_size=(2, 3), img_dim=(336, 500), path=None, num=0):
    if type(img_dim) is int:
        img_dim = (img_dim, img_dim)
    img_dim = tuple(img_dim)
    if len(img_dim) == 1:
        h_dim = img_dim
        w_dim = img_dim
    elif len(img_dim) == 2:
        h_dim, w_dim = img_dim
    h, w = figure_size
    if split is None:
        num_of_imgs = figure_size[0] * figure_size[1]
        gap = len(imgs) // num_of_imgs
        split = list(range(0, len(imgs) + 1, gap))
    figure = np.zeros((h_dim * h, w_dim * w, 3))
    for i in range(h):
        for j in range(w):
            idx = i * w + j
            if idx >= len(split) - 1:
                break
            digit = imgs[split[idx]: split[idx + 1]]
            if len(digit) == 1:
                for k in range(3):
                    figure[i * h_dim: (i + 1) * h_dim,
                    j * w_dim: (j + 1) * w_dim, k] = digit
            elif len(digit) == 3:
                for k in range(3):
                    figure[i * h_dim: (i + 1) * h_dim,
                    j * w_dim: (j + 1) * w_dim, k] = digit[2 - k]
    if path is None:
        cv2.imshow('Figure%d' % num, figure)
        cv2.waitKey()
    else:
        figure *= 255
        filename1 = path.split('\\')[-1]
        filename2 = path.split('/')[-1]
        if len(filename1) < len(filename2):
            filename = filename1
        else:
            filename = filename2
        root_path = path[:-len(filename)]
        if not os.path.exists(root_path):
            os.makedirs(root_path)
        log("Saving Image at {}".format(path))
        cv2.imwrite(path, figure)
