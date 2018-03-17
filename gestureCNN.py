from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import os
import theano
import json
import cv2
import matplotlib

from PIL import Image
from sklearn.utils import shuffle
from sklearn.model_selection import train_test_split
from matplotlib import pyplot as plt


minValue = 70

x0 = 400
y0 = 200
height = 600
width = 600



img_rows, img_cols = 100,100
img_channels = 1
batch_size = 32
nb_classes = 5
nb_epoch = 15
nb_filters = 32
nb_pool = 2
nb_conv = 3

path = "./"
path1 = "./gestures"
weight_file = ''

def loadCNN(wf_index):
    global get_output
    model = Sequential()
    
    
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv),
                        padding='valid',
                        input_shape=(img_channels, img_rows, img_cols)))
    convout1 = Activation('relu')
    model.add(convout1)
    model.add(Conv2D(nb_filters, (nb_conv, nb_conv)))
    convout2 = Activation('relu')
    model.add(convout2)
    model.add(MaxPooling2D(pool_size=(nb_pool, nb_pool)))
    model.add(Dropout(0.5))

    model.add(Flatten())
    model.add(Dense(128))
    model.add(Activation('relu'))
    model.add(Dropout(0.5))
    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))

    model.compile(loss='categorical_crossentropy', optimizer='adadelta', metrics=['accuracy'])
    
    
    
    model.summary()

    model.get_config()
    
    from keras.utils import plot_model
    plot_model(model, to_file='new_model.png', show_shapes = True)
    

    if wf_index >= 0:
        #Load pretrained weights
        fname = weight_file
        print "loading ", fname
        model.load_weights(fname)
    
    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    
    
    return model
