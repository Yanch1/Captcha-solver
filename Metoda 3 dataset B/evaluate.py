import os
from utils import gen_dataset, load_data, matprint, num_alphabet
from keras.models import load_model
from dotenv import dotenv_values
import cv2
import numpy as np
from tqdm import tqdm

config = dotenv_values('.env')

c1_correct = 0
c2_correct = 0
c3_correct = 0
c4_correct = 0
c5_correct = 0

total_correct = 0

correct_guesses_dict = {}

# Load config 
BATCH_SIZE = int(config['BATCH_SIZE'])
NUM_OF_LETTERS = int(config['NUM_OF_LETTERS'])
EPOCHS = int(config['EPOCHS'])
IMG_ROWS = int(config['IMG_ROWS'])
IMG_COLS = int(config['IMG_COLS'])

alphabet = list('qwertyuiopasdfghjklzxcvbnm0123456789')

# Non-configs
PATH = os.getcwd()
DATA_PATH = os.path.join(PATH, 'test_data')

model = load_model(os.path.join(PATH, "saved_models", "trained_model.h5"))

gen_dataset(DATA_PATH, 10 , NUM_OF_LETTERS, IMG_COLS, IMG_ROWS)

images = []
for root, directories, files in os.walk(DATA_PATH):
    for file in tqdm(files):
        img = cv2.imread(os.path.join(root, file))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        img = cv2.resize(img, (int(135/2), int(50/2)), interpolation=cv2.INTER_AREA)
        img = np.reshape(img, (img.shape[0], img.shape[1], 1))
        
        temp = [img]
        temp = np.array(temp)
        temp = temp.astype('float32')
        temp /= 255

        p = model.predict(temp)
        prediction = ""
        for a in p:
            for valid_list in a:
                valid_list = list(valid_list)
                idx = valid_list.index(max(valid_list))

                prediction += alphabet[idx]

        gc1, gc2, gc3, gc4, gc5 = prediction
        c1, c2, c3, c4, c5 = file.split(sep='_')[0]

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

        if prediction == ''.join([c1,c2,c3,c4,c5]):
            total_correct += 1

        n = correct_guesses_dict.get(correct_guesses, 0) + 1
        correct_guesses_dict[correct_guesses] = n
        
        print(f'prediction: {prediction}')
        print(f"actual content (filename): {file.split(sep='_')[0]}")
        print('-----------------------------------')

print(f"correct c1: {c1_correct}")
print(f"correct c2: {c2_correct}")
print(f"correct c3: {c3_correct}")
print(f"correct c4: {c4_correct}")
print(f"correct c5: {c5_correct}")

print(f"correct total: {total_correct}")

print(correct_guesses_dict)

        

        
