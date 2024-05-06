import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import random

def load_and_preprocess_image(path, im_size):
    # Load the image from disk
    img = tf.io.read_file(path)
    # Decode the image to a tensor
    img = tf.image.decode_image(img, channels=3)
    # Ensure the image is cast to a float32 data type
    img = tf.image.convert_image_dtype(img, tf.float32)
    # Resize the image to the desired size
    img = tf.image.resize(img, (im_size, im_size))
    return img

class LoadMyDataset(tf.keras.utils.Sequence):
    def __init__(self, edge_path, img_path, im_size=256, train=True):
        self.input_path = edge_path
        self.target_path = img_path
        self.im_size = 256
        self.input_list = [f for f in os.listdir(self.input_path) if f.endswith('.jpg')]
        self.target_list = [f for f in os.listdir(self.target_path) if f.endswith('.jpg')]

        if train:
            self.transform = self.preprocess_train_image
            self.num_images = 100
        else:
            self.transform = self.preprocess_test_image
            self.num_images = 100

    def preprocess_train_image(self, input_image, target_image, im_size):
        # Apply random resized crop augmentation
        input_image = tf.image.random_crop(input_image, size=[im_size, im_size, 3])
        input_image = tf.image.resize(input_image, [im_size, im_size])

        target_image = tf.image.random_crop(target_image, size=[im_size, im_size, 3])
        target_image = tf.image.resize(target_image, [im_size, im_size])
        return input_image, target_image

    def preprocess_test_image(self, input_image, target_image, im_size):
        # Resize the image for evaluation
        input_image = tf.image.resize(input_image, (im_size, im_size))
        target_image = tf.image.resize(target_image, (im_size, im_size))
        return input_image, target_image

    def __getitem__(self, item):
        input_name = self.input_list[item]
        target_name = self.target_list[item]
        
        input_path = os.path.join(self.input_path, input_name)
        target_path = os.path.join(self.target_path, target_name)
        
        input_image = load_and_preprocess_image(input_path, self.im_size)
        target_image = load_and_preprocess_image(target_path, self.im_size)
        
        # print("Values of variables:")
        # print("input_image shape:", input_name)
        # print("input_image shape:", input_image.shape)
        # print("target_image shape:", target_image.shape)
        input_image, target_image = self.transform(input_image, target_image, self.im_size)
        
        return input_image, target_image

    def __len__(self):
        return len(self.input_list)

    def load_name(self, index):
        name = self.input_list[index]
        return os.path.basename(name)

# Example usage:
edge_path = "data/images"
img_path = "data/sketches"
im_size = 256
train_dataset = LoadMyDataset(edge_path, img_path, im_size=im_size, train=True)

test_dataset = LoadMyDataset(edge_path, img_path, im_size=im_size, train=False)

# Randomize the order of training samples
random.shuffle(train_dataset.input_list)


# Loading the first 35,000 images for training
input_images_train, target_images_train = [], []
for i in tqdm(range(0, 99), desc="Loading Training Images"):
    try:
        input_image, target_image = train_dataset[i]
        input_images_train.append(input_image)
        target_images_train.append(target_image)
    except Exception as e:
        print(f"Failed to load images for file: {train_dataset.load_name(i)}")
        print(f"Error: {e}")

# Stack the input and target images into tensors for training
input_images_train = tf.stack(input_images_train)
np.save('data/input_images_train.npy', input_images_train.numpy())
del input_images_train

target_images_train = tf.stack(target_images_train)
np.save('data/target_images_train.npy', target_images_train.numpy())


del target_images_train
del train_dataset



# Loading the next 15,000 images for testing
input_images_test, target_images_test = [], []
for i in tqdm(range(100, 199), desc="Loading Testing Images"):
    try:
        input_image, target_image = test_dataset[i]
        input_images_test.append(input_image)
        target_images_test.append(target_image)
    except Exception as e:
        print(f"Failed to load images for file: {test_dataset.load_name(i)}")
        print(f"Error: {e}")



# Stack the input and target images into tensors for testing
input_images_test = tf.stack(input_images_test)
np.save('data/input_images_test.npy', input_images_test.numpy())
del input_images_test


target_images_test = tf.stack(target_images_test)
np.save('data/target_images_test.npy', target_images_test.numpy())

del target_images_test
del test_dataset
# Save input_images_train, target_images_train, input_images_test, and target_images_test as NumPy arrays




print("done-sies")

# # Display the first 5 images from the loaded dataset
# num_images_to_display = 5

# # Plot input and target images side by side
# fig, axes = plt.subplots(num_images_to_display, 4, figsize=(20, 20))

# for i in range(num_images_to_display):
#     # Plot input image for training dataset
#     axes[i, 0].imshow(input_images_train[i])
#     axes[i, 0].set_title('Train Input Image')
#     axes[i, 0].axis('off')

#     # Plot target image for training dataset
#     axes[i, 1].imshow(target_images_train[i])
#     axes[i, 1].set_title('Train Target Image')
#     axes[i, 1].axis('off')

#     # Plot input image for testing dataset
#     axes[i, 2].imshow(input_images_test[i])
#     axes[i, 2].set_title('Test Input Image')
#     axes[i, 2].axis('off')

#     # Plot target image for testing dataset
#     axes[i, 3].imshow(target_images_test[i])
#     axes[i, 3].set_title('Test Target Image')
#     axes[i, 3].axis('off')

# plt.tight_layout()
# plt.show()

# # Print filenames for training dataset
# print("Filenames for training dataset:")
# for i in range(35):
#     train_input_name = train_dataset.load_name(i)
#     train_target_name = train_dataset.load_name(i)
#     print(f"Train Input Image {i+1}: {train_input_name}")
#     print(f"Train Target Image {i+1}: {train_target_name}")

# # Print filenames for testing dataset
# print("\nFilenames for testing dataset:")
# for i in range(15):
#     test_input_name = test_dataset.load_name(i)
#     test_target_name = test_dataset.load_name(i)
#     print(f"Test Input Image {i+1}: {test_input_name}")
#     print(f"Test Target Image {i+1}: {test_target_name}")
