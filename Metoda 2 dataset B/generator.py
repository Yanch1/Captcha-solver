import os
import shutil
from captcha.image import ImageCaptcha
from tqdm import tqdm
from random import choices
import uuid

NUM_COLS = 190
NUM_ROWS = 80
DATASET_SIZE = 20000


alphabet = list('qwertyuiopasdfghjklzxcvbnm0123456789')
num_alphabet = len(alphabet)

DATA_PATH = os.path.join(os.getcwd(), 'dataset', 'raw')


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
    _gen_captcha(path, num_of_letters, num_of_repetition, width, height)
    print('Finished Data Generation')


gen_dataset(DATA_PATH, DATASET_SIZE, 5, NUM_COLS, NUM_ROWS)