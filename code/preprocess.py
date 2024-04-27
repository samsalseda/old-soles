import os
from PIL import Image
import numpy as np
import torchvision.transforms as transforms
import pickle
import random

def preprocess_data(input_path, target_path, input_list, target_list, transform):
    data = []
    for input_filename, target_filename in zip(input_list, target_list):
        input_img = Image.open(os.path.join(input_path, input_filename)).convert('RGB')
        target_img = Image.open(os.path.join(target_path, target_filename)).convert('RGB')

        # Apply transformations
        input_img = transform(input_img)
        target_img = transform(target_img)

        data.append((input_img, target_img))

    return data

def load_data(data_folder, im_size=256, shuffle=True):
    input_path = os.path.join(data_folder, 'images2')
    target_path = os.path.join(data_folder, 'sketches2')
    input_list = [f for f in os.listdir(input_path) if f.endswith('.jpg')]
    target_list = [f for f in os.listdir(target_path) if f.endswith('.jpg')]
    
    if shuffle:
        transform = transforms.Compose([
                transforms.RandomResizedCrop((im_size, im_size), scale=(0.9, 1.2)),
                transforms.ToTensor()
            ])
    else:
        transform = transforms.Compose([
                transforms.Resize((im_size, im_size)),
                transforms.ToTensor()
            ])

    return preprocess_data(input_path, target_path, input_list, target_list, transform)

def create_pickle(data_folder):
    shuffle_true_data = load_data(data_folder, shuffle=True)
    shuffle_false_data = load_data(data_folder, shuffle=False)

    with open(os.path.join(data_folder, 'data_shuffled_true.p'), 'wb') as pickle_file:
        pickle.dump(shuffle_true_data, pickle_file)
    print(f'Data with shuffle=True has been dumped into {os.path.join(data_folder, "data_shuffled_true.p")}!')

    with open(os.path.join(data_folder, 'data_shuffled_false.p'), 'wb') as pickle_file:
        pickle.dump(shuffle_false_data, pickle_file)
    print(f'Data with shuffle=False has been dumped into {os.path.join(data_folder, "data_shuffled_false.p")}!')

if __name__ == '__main__':
    data_folder = 'data'
    create_pickle(data_folder)
