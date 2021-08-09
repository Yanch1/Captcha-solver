import numpy as np
import random
import string
import sys
from pathlib import Path
from PIL import Image, ImageDraw, ImageFont
from captcha.image import ImageCaptcha


font = ImageFont.truetype('Vera.ttf', size=50)

path = "dataset/raw/"

def gen_string(size=5, chars=string.ascii_lowercase + string.digits):
    return ''.join(random.choice(chars) for _ in range(size))

if len(sys.argv) == 1:
    print("Specify number of CAPTCHA's to be generated.")
else:
    num = int(sys.argv[1])
    print(f"Generating {num} CAPTCHA's")

    image = ImageCaptcha(width=190, height=80)

    for i in range(num):
        if i % 10 == 0:
            print(f"{str(i)} CAPTCHA's generated")

        text = gen_string()

        filename = text + ".png"

        Path(path).mkdir(parents=True, exist_ok=True)

        image.write(text, path+filename)
        
    print("Finished")





