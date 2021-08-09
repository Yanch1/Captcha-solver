import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from utils import resize_to_fit, clear_chunks, stack_windows
from imutils import paths
import numpy as np
import imutils
import cv2 as cv2
import pickle
from pathlib import Path
from utils import resize_to_fit

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = Path('./test captchas/')
CAPTCHA_CHARACTERS_NUM = 5 #number of characters in captcha
COLUMN_WIDTH = 5
## raw: 280 x 96



# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)


i = 0
for image_file in paths.list_images(CAPTCHA_IMAGE_FOLDER):
    i+=1
    file_name = str(image_file).split(sep='\\')[2]
        
    prediction = ''

    image = cv2.imread(image_file)
    image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)

    image = cv2.resize(image, (133, 56), interpolation=cv2.INTER_AREA)
    
    

    for i in range(CAPTCHA_CHARACTERS_NUM):

        abi = cv2.copyMakeBorder(image, 0, 0, ((CAPTCHA_CHARACTERS_NUM - i - 1) * COLUMN_WIDTH), 0, cv2.BORDER_CONSTANT, value=0)
        abi = cv2.copyMakeBorder(abi,0,0,COLUMN_WIDTH,0, cv2.BORDER_CONSTANT, value=255)
        abi = cv2.copyMakeBorder(abi, 0, 0, (i * COLUMN_WIDTH), 0, cv2.BORDER_CONSTANT, value=0)

        # add dim
        abi = np.reshape(abi, (abi.shape[0], abi.shape[1], 1))

        data = np.array([abi], dtype="float32") / 255.0

        p = model.predict(data)

        p = lb.inverse_transform(p)[0]

        prediction = prediction + p
    
    print(f"image #{i} correct answer is: {file_name}")
    print(f"model's prediction is: {prediction}")