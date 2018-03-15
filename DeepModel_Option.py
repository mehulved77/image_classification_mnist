from __future__ import absolute_import, print_function
import os
import sys
import tensorflow
import pylab as pl
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import numpy as np
from pprint import pprint
import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten, Conv2D, MaxPooling2D

global args
from args import x_train, y_train, x_test, y_test, batch_size, num_classes, input_shape


np.set_printoptions(precision=5, suppress=True)
#%matplotlib inline

print('Python Version: {}'.format(sys.version_info[0]))
print('TensorFlow Version: {}'.format(tensorflow.__version__))
print('Keras Version: {}'.format(keras.__version__))
print('GPU Enabled?: {}'.format(tensorflow.test.gpu_device_name() is not ''))


# define the larger model
def larger_cnn_model():
    # create model
    model = Sequential()
    model.add(Conv2D(30, (5, 5), input_shape=input_shape, activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(15, (3, 3), activation='relu'))
    model.add(MaxPooling2D(pool_size=(2, 2))) 
    model.add(Dropout(0.2))
    model.add(Flatten())
    model.add(Dense(128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(num_classes, activation='softmax'))
    # Compile model
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model

large_cnn_model = larger_cnn_model()