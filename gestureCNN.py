from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D, ZeroPadding2D
from keras.optimizers import SGD,RMSprop,adam
from keras.utils import np_utils
from keras import backend as K
K.set_image_dim_ordering('th')

import numpy as np
import os
# import theano
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
nb_classes = 3
nb_epoch = 15
nb_filters = 32
nb_pool = 2
nb_conv = 3

path = "./"
path2 = "./gestures"
weight_file = 'trained_weights'

output = ['Pause', 'Up', 'Play']


def modlistdir(path):
    listing = os.listdir(path)
    retlist = []
    for name in listing:
        #This check is to ignore any hidden files/folders
        if name.startswith('.'):
            continue
        retlist.append(name)
    return retlist



def initializers():
    imlist = modlistdir(path2)
    
    image1 = np.array(Image.open(path2 +'/' + imlist[0]))
    
    m,n = image1.shape[0:2] 
    total_images = len(imlist) 
    
    immatrix = np.array([np.array(Image.open(path2+ '/' + images).convert('L')).flatten()
                         for images in imlist], dtype = 'f')
    

    
    print(immatrix.shape)
    
    input("Press any key")
    
    label=np.ones((total_images,),dtype = int)
    
   
    label[0:599]=0
    label[599:1136]=1
    label[1136:]=2    
    
    data,Label = shuffle(immatrix,label, random_state=2)
    train_data = [data,Label]
     
    (X, y) = (train_data[0],train_data[1])
     
     
    # Split X and y into training and testing sets
     
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=4)
     
    X_train = X_train.reshape(X_train.shape[0], img_channels, img_rows, img_cols)
    X_test = X_test.reshape(X_test.shape[0], img_channels, img_rows, img_cols)
     
    X_train = X_train.astype('float32')
    X_test = X_test.astype('float32')
     
    # normalize
    X_train /= 255
    X_test /= 255
     
    
    Y_train = np_utils.to_categorical(y_train, nb_classes)
    Y_test = np_utils.to_categorical(y_test, nb_classes)
    return X_train, X_test, Y_train, Y_test



def trainModel(model):

    # Split X and y into training and testing sets
    X_train, X_test, Y_train, Y_test = initializers()

    # Now start the training of the loaded model
    hist = model.fit(X_train, Y_train, batch_size=batch_size, epochs=nb_epoch,
                 verbose=1, validation_split=0.2)

    visualizeHis(hist)

    
    filename = weight_file
    fname = path + str(filename) + ".hdf5"
    model.save_weights(fname,overwrite=True)
    

def visualizeHis(hist):
    train_loss=hist.history['loss']
    val_loss=hist.history['val_loss']
    train_acc=hist.history['acc']
    val_acc=hist.history['val_acc']
    xc=range(nb_epoch)

    plt.figure(1,figsize=(7,5))
    plt.plot(xc,train_loss)
    plt.plot(xc,val_loss)
    plt.xlabel('num of Epochs')
    plt.ylabel('loss')
    plt.title('train_loss vs val_loss')
    plt.grid(True)
    plt.legend(['train','val'])
    #print plt.style.available # use bmh, classic,ggplot for big pictures
    #plt.style.use(['classic'])

    plt.figure(2,figsize=(7,5))
    plt.plot(xc,train_acc)
    plt.plot(xc,val_acc)
    plt.xlabel('num of Epochs')
    plt.ylabel('accuracy')
    plt.title('train_acc vs val_acc')
    plt.grid(True)
    plt.legend(['train','val'],loc=4)

    plt.show()

                  

def loadCNN():
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
    

    # fname = weight_file
    # print("loading ", fname)
    # model.load_weights(fname)
    
    layer = model.layers[11]
    get_output = K.function([model.layers[0].input, K.learning_phase()], [layer.output,])
    
    model.get_weights()
    model.save_weights(path+str(weight_file))
    return model


def guessGesture(model, img):
    global output, get_output
    #Load image and flatten it
    image = np.array(img).flatten()
    
    # reshape it
    image = image.reshape(img_channels, img_rows,img_cols)
    
    # float32
    image = image.astype('float32') 
    
    # normalize it
    image = image / 255
    
    # reshape for NN
    rimage = image.reshape(1, img_channels, img_rows, img_cols)
    
    # Now feed it to the NN, to fetch the predictions
    #index = model.predict_classes(rimage)
    #prob_array = model.predict_proba(rimage)
    
    prob_array = get_output([rimage, 0])[0]
    
    #print prob_array
    
    d = {}
    i = 0
    for items in output:
        d[items] = prob_array[0][i] * 100
        i += 1
    print(d)
    print(output)
    # Get the output with maximum probability
    import operator
    
    guess = max(d.items(), key=operator.itemgetter(1))[0]
    prob  = d[guess]

    if prob > 70.0:
        #print guess + "  Probability: ", prob

        #Enable this to save the predictions in a json file,
        #Which can be read by plotter app to plot bar graph
        #dump to the JSON contents to the file
        
        with open('gesturejson.txt', 'w') as outfile:
            json.dump(d, outfile)

        return output.index(guess)

    else:
        return 1


