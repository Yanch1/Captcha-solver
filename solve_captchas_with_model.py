from keras.models import load_model
from utils import resize_to_fit, clear_chunks, stack_windows
from imutils import paths
import numpy as np
import imutils
import cv2 as cv2
import pickle
import os


MODEL_FILENAME = "captcha_model.hdf5"
MODEL_LABELS_FILENAME = "model_labels.dat"
CAPTCHA_IMAGE_FOLDER = "test captchas"


# Load up the model labels (so we can translate model predictions to actual letters)
with open(MODEL_LABELS_FILENAME, "rb") as f:
    lb = pickle.load(f)

# Load the trained neural network
model = load_model(MODEL_FILENAME)


for root, dirs, files in os.walk(CAPTCHA_IMAGE_FOLDER):
    for name in files:
        winnames = []

        kernel = (5,5)

        #load image
        image = cv2.imread(os.path.join(root, name))
        image = cv2.cvtColor(image, cv2.COLOR_RGBA2GRAY)
        cv2.imshow("a", image)
        winnames.append('a')

        #add padding
        image = cv2.copyMakeBorder(image, 8, 8, 8, 8, cv2.BORDER_CONSTANT, None, 255)
        cv2.imshow("b", image)
        winnames.append('b')


        #blur
        k = np.ones((5,5),np.float32)/25
        image = cv2.filter2D(image,-1,k)
        cv2.imshow("c", image)
        winnames.append('c')

        # threshhold image
        ret, image = cv2.threshold(image, 127, 255, cv2.THRESH_BINARY_INV)
        cv2.imshow("d", image)
        winnames.append('d')

        # clear white dots
        clear_chunks(image,0,50)
        cv2.imshow("e", image)
        winnames.append('e')

        # erosion
        image = cv2.erode(image, kernel, iterations=1)
        cv2.imshow("f", image)
        winnames.append('f')
        
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
            # This contour is too wide to be a single letter!
            # Split it in half into two letter regions!
                half_width = int(w / 2)
                letter_image_regions.append((x, y, half_width, h))
                letter_image_regions.append((x + half_width, y, half_width, h))
            else:
                letter_image_regions.append((x, y, w, h))

        if len(letter_image_regions) != 5:
            print(f"Found {len(letter_image_regions)} letter regions instead of 5 , skipping image")
            # continue
        
        letter_image_regions = sorted(letter_image_regions, key=lambda x: x[0])

        winnames2 = []
        chars = []
        i=0
        for (x,y,w,h) in letter_image_regions:
            letter = image[y-2:y+h+2, x-2:x+w+2]
            cv2.imshow(str(i),letter)
            winnames2.append(str(i))
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


      
            

        print(predictions)

        stack_windows(winnames)
        stack_windows(winnames2, height=200)
        cv2.waitKey(0)
        cv2.destroyAllWindows()