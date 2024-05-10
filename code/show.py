import numpy as np
import matplotlib.pyplot as plt


def print_images_from_npz(file_path):
    """
    Makes an images from np.

    :param file_path: file path to look for the image at
    """
    # Load the numpy file
    images_array = np.load(file_path)

    # Plot each image
    for i, image in enumerate(images_array):
        plt.imshow(image)
        plt.title(f"Image {i+1}")
        plt.axis("off")
        plt.show()


if __name__ == "__main__":  # example usage
    file_path = "target_images_train.npy"
    print_images_from_npz(file_path)
