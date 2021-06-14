from captcha.image import ImageCaptcha
import random
import imutils
from itertools import groupby, tee, cycle
from cv2 import cv2
import numpy as np
import os
from pathlib import Path
from utils import *


################################################################################
# preprocessing
################################################################################
path = r".\dataset\raw"
save_path = r".\dataset\letters"
letters_count = {}
skipped_images = 0


for root, dirs, files in os.walk(path):
    for name in files:

        kernel = (5,5)

        #load image
        image = cv2.imread(os.path.join(root, name))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

        #add padding
        image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, None, 255)

        #blur
        k = np.ones((5,5),np.float32)/25
        image = cv2.filter2D(image,-1,k)

        # threshhold image
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)

        # clear white dots
        clear_chunks(image,0,50)

        # erosion
        image = cv2.erode(image, kernel, iterations=1)

        # get contours
        contours = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)[-2]

        #segment letters
        letter_image_regions = [] #(x, y, w ,h)

        for contour in contours:
            
            if cv2.contourArea(contour) < 30:
                continue

            
            (x, y, w, h) = cv2.boundingRect(contour)

            if w / h > 1.2:
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))

        if len(letter_image_regions) != 5:
            print(f"Found {len(letter_image_regions)} letter regions instead of 5 , skipping image")
            skipped_images += 1
            continue

        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        chars = []
        i = 0
        for (x,y,w,h) in letter_image_regions:
            letter = image[y-2:y+h+2, x-2:x+w+2]
            chars.append((letter, name[i]))
            i+=1


        for letter in chars:
            Path(os.path.join(save_path, letter[1])).mkdir(parents=True, exist_ok=True)
            count = letters_count.get(letter[1], 1)
            cv2.imwrite(os.path.join(save_path, letter[1], str(count)+'.png'), letter[0])
            letters_count[letter[1]] = count + 1

print(f"Total skipped images: {skipped_images}")