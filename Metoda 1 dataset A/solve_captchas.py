import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'

from keras.models import load_model
from utils import resize_to_fit, clear_chunks, stack_windows
from imutils import paths
import numpy as np
import imutils
import cv2 as cv2
import pickle
from tqdm import tqdm

c1_correct = 0
c2_correct = 0
c3_correct = 0
c4_correct = 0
c5_correct = 0

total_correct = 0
incorrectly_segmented = 0

correct_guesses_dict = {}

MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "test captchas"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)


for root, dirs, files in os.walk(CAPTCHA_IMAGE_FOLDER):
    for name in tqdm(files, desc='Solving captchas'):
        
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
        
        
        contours = sorted(contours, key=lambda x: cv2.contourArea(x), reverse=True)
        contours = contours[:5]
        
        for contour in contours:
            
            if cv2.contourArea(contour) < 60:
                continue

            
            (x, y, w, h) = cv2.boundingRect(contour)

            if w / h > 1.5:
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))

        if len(letter_image_regions) != 5:
            incorrectly_segmented += 1
            continue
            print(f"Found {len(letter_image_regions)} letter regions instead of 5 , the guess will likely be incorrect")
            
        
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        chars = []
        i=0
        for (x,y,w,h) in letter_image_regions:
            letter = image[y-2:y+h+2, x-2:x+w+2]
            chars.append(letter)
            i+=1

        predictions = []

        for letter in chars:
            # Re-size the letter image to 20x20 pixels to match training data
            letter = resize_to_fit(letter, 20, 20)

            # Turn the single image into a 4d list of images to make Keras happy
            letter = np.expand_dims(letter, axis=2)
            letter = np.expand_dims(letter, axis=0)

            # Ask the neural network to make a prediction
            prediction = model.predict(letter)

            # Convert the one-hot-encoded prediction back to a normal letter
            letter_text = lb.inverse_transform(prediction)[0]
            predictions.append(letter_text)

        gc1, gc2, gc3, gc4, gc5 = predictions
        c1, c2, c3, c4, c5, e1, e2, e3, e4 = name 

        correct_guesses = 0

        if c1 == gc1:
            c1_correct += 1
            correct_guesses += 1
        if c2 == gc2:
            c2_correct += 1
            correct_guesses += 1
        if c3 == gc3:
            c3_correct += 1
            correct_guesses += 1
        if c4 == gc4:
            c4_correct += 1
            correct_guesses += 1
        if c5 == gc5:
            c5_correct += 1
            correct_guesses += 1

        if ''.join(predictions) == ''.join([c1,c2,c3,c4,c5]):
            total_correct += 1

        n = correct_guesses_dict.get(correct_guesses, 0) + 1
        correct_guesses_dict[correct_guesses] = n

        print(f"Prediction for {name}: {''.join(predictions)}")
    
print(f"correct c1: {c1_correct}")
print(f"correct c2: {c2_correct}")
print(f"correct c3: {c3_correct}")
print(f"correct c4: {c4_correct}")
print(f"correct c5: {c5_correct}")

print(f"correct total: {total_correct}")

print(f"correctly segmented: {10000 - incorrectly_segmented}")

print(correct_guesses_dict)
            
        
        