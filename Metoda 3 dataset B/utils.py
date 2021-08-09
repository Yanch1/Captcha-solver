import os
import shutil
from captcha.image import ImageCaptcha
from random import choices, random
import numpy as np
import uuid
from tqdm import tqdm
import cv2

from dotenv import dotenv_values

###############################################
# Load config
###############################################
config = dotenv_values('.env')

BATCH_SIZE = int(config['BATCH_SIZE'])
NUM_OF_LETTERS = int(config['NUM_OF_LETTERS'])
EPOCHS = int(config['EPOCHS'])
IMG_ROWS = int(config['IMG_ROWS'])
IMG_COLS = int(config['IMG_COLS'])

alphabet_all = list('qwertyupasdfghjkzxcvbnm23456789QWERTYUPKJHGFDSAZXCVBNM')
alphabet = list('qwertyuiopasdfghjklzxcvbnm0123456789')
num_alphabet = len(alphabet)



###############################################
# Utility functions
###############################################
def _gen_captcha(img_dir, num_of_letters, num_of_repetition, width, height):

    #delete dir if exists and create it again
    if os.path.exists(img_dir):
        shutil.rmtree(img_dir)
    if not os.path.exists(img_dir):
        os.makedirs(img_dir)

    image = ImageCaptcha(width=width, height=height)

    for counter in tqdm(range(num_of_repetition), desc='Generating CAPTCHA\'S'):
        letters = choices(alphabet, k=5)
        captcha_text = ''.join(letters)
        path = os.path.join(img_dir, f"{captcha_text}_{uuid.uuid4()}.png")
        image.write(captcha_text, path)


def gen_dataset(path, num_of_repetition, num_of_letters, width, height):
    _gen_captcha(os.path.join(path, 'data'), num_of_letters, num_of_repetition, width, height)
    print('Finished Data Generation')


def load_data(path, test_split=0.1):
    print ('loading dataset...')
    y_train = []
    y_test = []
    x_train = []
    x_test = []

    # r=root, d=directories, f = files          flr = a3js2   _3234214er2q14e12.png
    counter = 0
    for root, directories, files in os.walk(path):
        for file in tqdm(files, desc='Loading and preprocessing the data'):
            if '.png' in file:
                counter += 1
                captcha_text = file.split('_')[0]
                label = np.zeros((NUM_OF_LETTERS, num_alphabet))
                for i in range(NUM_OF_LETTERS):
                    label[i, alphabet.index(captcha_text[i].lower())] = 1  # generate one hot encoded label

                img = cv2.imread(os.path.join(root, file))
                img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
                img = cv2.resize(img, (int(135/2), int(50/2)), interpolation=cv2.INTER_AREA)
                img = np.reshape(img, (img.shape[0], img.shape[1], 1))

                if random() < test_split:
                    y_test.append(label)
                    x_test.append(img)
                else:
                    y_train.append(label)
                    x_train.append(img)

    print('Finished loading and preprocessing the data')
    print('dataset size:', counter, f'(train={len(y_train)}, test={len(y_test)})')
    return np.array(x_train), np.array(y_train), np.array(x_test), np.array(y_test)


def matprint(mat, fmt="g"):
    col_maxes = [max([len(("{:"+fmt+"}").format(x)) for x in col]) for col in mat.T]
    for x in mat:
        for i, y in enumerate(x):
            print(("{:"+str(col_maxes[i])+fmt+"}").format(y), end="  ")
        print("")


