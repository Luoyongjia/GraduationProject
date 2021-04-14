import time
import numpy as np

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
