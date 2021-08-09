from captcha.image import ImageCaptcha
import random
import imutils
from itertools import groupby, tee, cycle
from cv2 import cv2
import numpy as np
import os
from pathlib import Path
import imutils


###################################
# utility functions
###################################

def stack_windows(win_names, height=0):
    x = 0
    for i in range(len(win_names)):
    
        cv2.moveWindow(win_names[i], x, height)

        rect = cv2.getWindowImageRect(win_names[i])
        x = x + rect[2]


def find_sequences(src):
    x2 = cycle(src)
    next(x2)
    grps = groupby(src, key=lambda j: j + 1 == next(x2))
    for k, v in grps:
        if k:
            yield tuple(v) + (next((next(grps)[1])),)


def white_pixels_in_image(img):
    black, white = (0,0)
    for h in range(img.shape[0]):
        for w in range(img.shape[1]):
            if not img[h][w]:
                black+=1
            else:
                white+=1
    if black+white == 0:
        return 0
    else:
        return (white / (black + white)) * 100


def clear_chunks(image, min_size, max_size):
    cnts = cv2.findContours(image, cv2.RETR_LIST, cv2.CHAIN_APPROX_SIMPLE)[-2]

    s1 = min_size
    s2 = max_size

    for cnt in cnts:
        if s1 < cv2.contourArea(cnt) < s2:
            cv2.fillPoly(image,pts=[cnt], color = (0,0,0))


def resize_to_fit(image, width, height):
    """
    A helper function to resize an image to fit within a given size
    :param image: image to resize
    :param width: desired width in pixels
    :param height: desired height in pixels
    :return: the resized image
    """

    # grab the dimensions of the image, then initialize
    # the padding values
    (h, w) = image.shape[:2]

    # if the width is greater than the height then resize along
    # the width
    if w > h:
        image = imutils.resize(image, width=width)

    # otherwise, the height is greater than the width so resize
    # along the height
    else:
        image = imutils.resize(image, height=height)

    # determine the padding values for the width and height to
    # obtain the target dimensions
    padW = int((width - image.shape[1]) / 2.0)
    padH = int((height - image.shape[0]) / 2.0)

    # pad the image then apply one more resizing to handle any
    # rounding issues
    image = cv2.copyMakeBorder(image, padH, padH, padW, padW,
        cv2.BORDER_REPLICATE)
    image = cv2.resize(image, (width, height))

    # return the pre-processed image
    return image