import numpy as np
from PIL import Image
import os
from preprocess_temp import process

def save_image(img_array:np.array, name:str):
    output_dir = "code/outputs/"
    os.makedirs(output_dir, exist_ok=True)
    img = Image.fromarray(img_array*255, 'RGB')
    img.save(output_dir + name + ".png", 'PNG')

# if __name__ == "__main__":
#     inputs, outputs = process()
#     save_image(np.asarray(inputs[0]), "testing")