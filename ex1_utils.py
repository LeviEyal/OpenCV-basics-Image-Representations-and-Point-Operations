from typing import List

import numpy as np
import matplotlib.pyplot as plt
import cv2
import math

LOAD_GRAY_SCALE = 1
LOAD_RGB = 2
RGB_TO_YIQ_MATRIX = np.array([[0.299, 0.587, 0.114],
                               [0.596, -0.275, -0.321],
                               [0.212, -0.523, 0.311]])
def myID() -> np.int:
    """
    Return my ID
    """
    return 203249073


def imReadAndConvert(filename: str, representation: int) -> np.ndarray:
    """
    Reads an image, and returns the image converted as requested
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: The image object
    """
    img = cv2.imread(filename).astype(np.float32)/255
    rep = cv2.COLOR_BGR2RGB if representation == LOAD_RGB else cv2.COLOR_BGR2GRAY
    return cv2.cvtColor(img, rep)


def imDisplay(filename: str, representation: int):
    """
    Reads an image as RGB or GRAY_SCALE and displays it
    :param filename: The path to the image
    :param representation: GRAY_SCALE or RGB
    :return: None
    """
    img = imReadAndConvert(filename, representation)                    # Read the image
    img = (img * 255).astype(np.uint8)                                  # Normalize to [0, 255]
    plt.gray()                                                          # Set the plot to grayscale
    plt.imshow(img)
    plt.show()


def transformRGB2YIQ(imgRGB: np.ndarray) -> np.ndarray:
    """
    Converts an RGB image to YIQ color space
    :param imgRGB: An Image in RGB
    :return: A YIQ in image color space
    """
    return imgRGB.dot(RGB_TO_YIQ_MATRIX)


def transformYIQ2RGB(imgYIQ: np.ndarray) -> np.ndarray:
    """
    Converts an YIQ image to RGB color space
    :param imgYIQ: An Image in YIQ
    :return: A RGB in image color space
    """
    return imgYIQ.dot(np.linalg.inv(RGB_TO_YIQ_MATRIX))


def hsitogramEqualize(imgOrig: np.ndarray) -> (np.ndarray, np.ndarray, np.ndarray):
    """
        Equalizes the histogram of an image
        :param imgOrig: Original Histogram
        :return: (imgEq,histOrg,histEQ)
    """
    imgOrig = (imgOrig * 255).astype(np.uint8)                          # Normalize to [0, 255]
    histOrg = np.histogram(imgOrig, bins=256, range=(0, 255))[0]        # Get original histogram
    cum_sum = np.cumsum(histOrg)                                        # Get cumulative sum
    lut = np.floor((cum_sum / cum_sum.max()) * 255)                     # Get equalized LUT
    imgEq = np.array([[lut[i] for i in row] for row in imgOrig])        # Apply LUT
    histEQ = np.histogram(imgEq, bins=256, range=(0, 255))[0]           # Get equalized histogram
    return (imgEq / 255.0), histOrg, histEQ


def quantizeImage(imOrig: np.ndarray, nQuant: int, nIter: int) -> (List[np.ndarray], List[float]):
    """
        Quantized an image in to **k** colors
        :param imOrig: The original image (RGB or Gray scale)
        :param nQuant: Number of colors to quantize the image to
        :param nIter: Number of optimization loops
        :return: (List[qImage_i],List[error_i])
    """
    data = imOrig.copy() if len(imOrig.shape) != 3 else transformRGB2YIQ(imOrig)[:, :, 0]
    data = np.dot(data, 255).astype(int)  # De-Normalize and convert to int

    histOrg, binsOrig = np.histogram(data, np.arange(257))
    cumsum = np.cumsum(histOrg)
    bounds = [0]
    n = 1
    for idx, value in enumerate(cumsum):
        if value / cumsum[-1] > n / nQuant:
            bounds.append(idx)
            n += 1
    bounds.append(255)

    imgs = []
    MSE = []
    ALL_PIXELS = histOrg.sum()

    for itr in range(nIter):
        q = []
        for k in range(len(bounds) - 1):
            start = bounds[k]
            end = bounds[k + 1]
            if k == (len(bounds) - 2):
                end += 1
            if np.asarray(histOrg[start:end]).sum() != 0:
                q_i = round((np.asarray((binsOrig[start:end] * histOrg[start:end])).sum())
                            / np.asarray(histOrg[start:end]).sum())
            else:
                q_i = 0
            q.append(q_i)

        quantized = data.copy()

        for i in range(len(bounds) - 1):
            quantized = np.where(((bounds[i] <= quantized) & (quantized < bounds[i + 1])),
                                        q[i], quantized)
        if len(imOrig.shape) == 3:
            yiq_image = transformRGB2YIQ(imOrig).dot(255).astype(int)
            yiq_image[:, :, 0] = quantized
            imgQu = transformYIQ2RGB(yiq_image).astype(int)
        else:
            imgQu = quantized

        imgs.append(imgQu)

        # Add the MSE for this iteration to the list
        MSE.append((math.sqrt(((quantized - data) ** 2).sum())) / ALL_PIXELS)

        bounds = [0]
        for i in range(len(q) - 1):
            bounds.append(int((q[i] + q[i + 1]) / 2))
        bounds.append(255)

    return imgs, MSE
