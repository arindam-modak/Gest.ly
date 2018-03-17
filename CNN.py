from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils

# We require this for Theano lib ONLY. Remove it for TensorFlow usage
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
#import matplotlib.pyplot as plt
import os
import theano
from PIL import Image
# SKLEARN
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
import json

import cv2
import matplotlib
#matplotlib.use("TkAgg")
from matplotlib import pyplot as plt

img_rows, img_cols = 200, 200

batch_size = 32

nb_classes = 5

nb_epoch = 15

nb_filters = 32

nb_pool = 2

nb_conv = 3

path = "./"

path2 = "./gestures"

output = ["PLAY", "PAUSE", "UP", "NOTHING"]
