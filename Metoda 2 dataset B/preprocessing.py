import random
import imutils
from itertools import groupby, tee, cycle
from cv2 import cv2
import numpy as np
import os
from pathlib import Path

from utils import *

from tqdm import tqdm

################################################################################
# preprocessing
################################################################################
path = Path("dataset/raw/")
save_path = Path("dataset/abi/")

CAPTCHA_CHARACTERS_NUM = 5 #number of characters in captcha
COLUMN_WIDTH = 5

chars = {}

Path(os.path.join(save_path)).mkdir(parents=True, exist_ok=True)

for root, dirs, files in os.walk(path):
    for name in tqdm(files, desc='Preprocessing images'):
        #load image
        image = cv2.imread(os.path.join(root, name))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        
        for i in range(CAPTCHA_CHARACTERS_NUM):

            abi = cv2.copyMakeBorder(image, 0, 0, ((CAPTCHA_CHARACTERS_NUM - i - 1) * COLUMN_WIDTH), 0, cv2.BORDER_CONSTANT, value=0)
            abi = cv2.copyMakeBorder(abi,0,0,COLUMN_WIDTH,0, cv2.BORDER_CONSTANT, value=255)
            abi = cv2.copyMakeBorder(abi, 0, 0, (i * COLUMN_WIDTH), 0, cv2.BORDER_CONSTANT, value=0)
            
            n = chars.get(name[i], 0)
            file = save_path / (name[i] + '_' + str(n) + '.png')
            
            chars[name[i]] = n + 1

            cv2.imwrite(str(file), abi)
            
