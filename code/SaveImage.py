import numpy as np
from PIL import Image

def save_image(img_array:np.array, name:str):
    img = Image.fromarray(img_array, 'RGB')
    img.save("outputs/" + name + ".png", 'PNG')

