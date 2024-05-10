import numpy as np
from PIL import Image
import os
from pp import load_and_preprocess_image
import tensorflow as tf
import matplotlib.pyplot as plt


def save_image(img_array: np.array, name: str):
    """
    Saves an image.

    :param img_array: numpy array to save
    :param name: name to save the image as
    """
    print(img_array)
    output_dir = "code/outputs/"
    os.makedirs(output_dir, exist_ok=True)
    img_array = np.array(img_array * 255, dtype=np.uint8)
    img = Image.fromarray(img_array, "RGB")

    plt.imshow(img)
    plt.show()

    img = np.array(img_array * 255, dtype=np.uint8)
    img = Image.fromarray(img_array, "RGB")
    img.save(output_dir + name + ".png", "PNG")


if __name__ == "__main__":
    path = "code/results/201.jpg"
    img_tensor = load_and_preprocess_image(path, 128)
    save_image(tf.make_ndarray(tf.make_tensor_proto(img_tensor)), "testing")
