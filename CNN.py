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

path = "./imgs"

path2 = "./gestures"

output = ["PLAY", "PAUSE", "UP"]

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
        fname = WeightFileName[int(wf_index)]
        print "loading ", fname
        model.load_weights(fname)

    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])


    return model

def guessGesture(model, img):
    global output, get_output

    image = np.array(img).flatten()


    image = image.reshape(img_channels, img_rows,img_cols)


    image = image.astype('float32')


    image = image / 255


    rimage = image.reshape(1, img_channels, img_rows, img_cols)


    prob_array = get_output([rimage, 0])[0]


    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1

    import operator

    guess = max(d.iteritems(), key=operator.itemgetter(1))[0]
    prob  = d[guess]

    if prob > 70.0:
        with open('gesturejson.txt', 'w') as outfile:
            json.dump(d, outfile)

        return output.index(guess)

    else:
        return 1
