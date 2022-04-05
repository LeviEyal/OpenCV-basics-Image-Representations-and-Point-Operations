from ex1_utils import imReadAndConvert
from ex1_utils import LOAD_GRAY_SCALE
import cv2
import numpy as np


def gammaDisplay(img_path: str, rep: int):
    """
    GUI for gamma correction
    :param img_path: Path to the image
    :param rep: grayscale(1) or RGB(2)
    :return: None
    """
    title_window = 'Gamma Correction'
    cv2.namedWindow(title_window)
    img = imReadAndConvert(img_path, rep)                                   # Read the image

    def trackbar_callback(gamma):
        img2 = np.power(img, gamma/100)                                     # gamma correction
        to_show = cv2.cvtColor(
            (img2 * 255).astype(np.uint8), cv2.COLOR_RGB2BGR)               # convert to BGR
        cv2.imshow(title_window, to_show)                                   # display

    cv2.createTrackbar('Gamma', title_window, 100, 200, trackbar_callback)  # create trackbar
    
    trackbar_callback(100)
    cv2.waitKey()
    cv2.destroyAllWindows()


def main():
    gammaDisplay('bac_con.png', LOAD_GRAY_SCALE)


if __name__ == '__main__':
    main()
