import numpy as np
import os
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from PIL import Image
from tqdm import tqdm
import random

def load_and_preprocess_image(path, im_size):
    '''
    Method used to handle basic image. Takes in the file path to a jpg as well 
    as the required size for the model (256 in our case). Converts the image
    into a tensor and returns. 
    '''
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
    '''
    Class designed to house all methods necessary to properly load thousands of
    images at a time given the necessary inputs. Here images are treated 
    slightly differently depending on their purpose. Mimicking the paper, images
    for training also have a random crop applied. These transformations are tied
    to the get function which ensures they are uniformly applied.  
    '''
    def __init__(self, edge_path, img_path, im_size=256, train=True):
        # defines the necessary properities of the dataset
        self.input_path = edge_path
        self.target_path = img_path
        self.im_size = 256
        self.input_list = [f for f in os.listdir(self.input_path) if f.endswith('.jpg')]
        self.target_list = [f for f in os.listdir(self.target_path) if f.endswith('.jpg')]

        #sets quantity of images as well as type of transformation. 
        if train:
            self.transform = self.preprocess_train_image
            self.num_images = 5000
        else:
            self.transform = self.preprocess_test_image
            self.num_images = 25

    # applies transformations for training image
    def preprocess_train_image(self, input_image, target_image, im_size):
        # Apply random resized crop augmentation
        input_image = tf.image.random_crop(input_image, size=[im_size, im_size, 3])
        input_image = tf.image.resize(input_image, [im_size, im_size])

        target_image = tf.image.random_crop(target_image, size=[im_size, im_size, 3])
        target_image = tf.image.resize(target_image, [im_size, im_size])
        return input_image, target_image
    
    # applies transformations for test image (lacks crop)
    def preprocess_test_image(self, input_image, target_image, im_size):
        # Resize the image for evaluation
        input_image = tf.image.resize(input_image, (im_size, im_size))
        target_image = tf.image.resize(target_image, (im_size, im_size))
        return input_image, target_image

    #custom get method applies the required preprocessing
    def __getitem__(self, item):
        input_name = self.input_list[item]
        target_name = self.target_list[item]
        
        input_path = os.path.join(self.input_path, input_name)
        target_path = os.path.join(self.target_path, target_name)
        
        input_image = load_and_preprocess_image(input_path, self.im_size)
        target_image = load_and_preprocess_image(target_path, self.im_size)

        input_image, target_image = self.transform(input_image, target_image, self.im_size)
        
        return input_image, target_image

    #returns length
    def __len__(self):
        return len(self.input_list)

    #returns name of file
    def load_name(self, index):
        name = self.input_list[index]
        return os.path.basename(name)

# Initializing for our data:
edge_path = "data/images"
img_path = "data/sketches"
im_size = 256

#Initializing both data sets for training and testing
train_dataset = LoadMyDataset(edge_path, img_path, im_size=im_size, train=True)
test_dataset = LoadMyDataset(edge_path, img_path, im_size=im_size, train=False)


# Randomize the order of training samples
random.shuffle(train_dataset.input_list)


# Loading the first x images for training
input_images_train, target_images_train = [], []
for i in tqdm(range(0, 1000), desc="Loading Training Images"):
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
#deletes old object for memory efficiency
del input_images_train

target_images_train = tf.stack(target_images_train)
np.save('data/target_images_train.npy', target_images_train.numpy())

#deletes old object for memory efficiency
del target_images_train
del train_dataset



# Loading the next 15,000 images for testing
input_images_test, target_images_test = [], []
for i in tqdm(range(15000, 15024), desc="Loading Testing Images"):
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
#deletes old object for memory efficiency
del input_images_test


target_images_test = tf.stack(target_images_test)
np.save('data/target_images_test.npy', target_images_test.numpy())

#deletes old objects for memory efficiency
del target_images_test
del test_dataset