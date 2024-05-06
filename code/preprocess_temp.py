import os
import numpy as np
import glob as glob
from imageio import imread


def process():
    i = 0
    filelist_input = glob.glob(os.path.join("./results", "*"))
    filelist_output = glob.glob(os.path.join("./sketches", "*"))
    inputs = []
    outputs = []
    for file_input, file_output in zip(sorted(filelist_input), sorted(filelist_output)):
        # print(i)
        # do some fancy stuff
        # print str(infile)
        vectorized_picture_input = np.array(imread(file_input)) / 255
        vectorized_picture_output = np.array(imread(file_output)) / 255
        inputs += [vectorized_picture_input]
        outputs += [vectorized_picture_output]
        # print(vectorized_picture_input[60])
        i += 1
        if i > 8:
            return np.asarray(inputs), np.asarray(outputs)
