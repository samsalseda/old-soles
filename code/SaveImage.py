import numpy as np
from PIL import Image
import os
#from preprocess_temp import process
from pp import load_and_preprocess_image
import tensorflow as tf
import matplotlib.pyplot as plt
import matplotlib.image as mpimg


def save_image(img_array:np.array, name:str):
    print(img_array)
    output_dir = "code/outputs/"
    os.makedirs(output_dir, exist_ok=True)
    img_array = np.array(img_array * 255, dtype=np.uint8)
    img = Image.fromarray(img_array, 'RGB')

    imgplot = plt.imshow(img)
    plt.show()

    img.save(output_dir + name + ".png", 'PNG')

<<<<<<< HEAD

if __name__ == "__main__":
    path = "code/results/201.jpg"
    img_tensor = load_and_preprocess_image(path, 128)
    save_image(tf.make_ndarray(tf.make_tensor_proto(img_tensor)), "testing")

    # if __name__ == "__main__":
#     inputs, outputs = process()
#     print(inputs[0].dtype)
#     save_image(tf.make_ndarray(tf.make_tensor_proto(inputs[0])), "testing")
=======
# if __name__ == "__main__":
#     inputs, outputs = process()
#     save_image(np.asarray(inputs[0]), "testing")
>>>>>>> 4f887af82da87e9fa92fe7b6f61df7c932c3c887
