#!/usr/bin/env python3

from keras.datasets import cifar10      # Subroutines for getting the CIFAR-10 data set
from keras.models import Model          # Class for specifying and training a Neural Network
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import np_utils        # Utilities for one-hot encoding of ground truth values
import numpy as np

batch_size = 32         # Number of training examples to consider in each iteration
num_epochs = 300        # Number of times to iterate over tge entire training set
kernel_size = 3         # Kernel size to be used throughout
pool_size = 2           # Pooling size to be used throughout
conv_depth_1 = 32       # Initial number of kernels in the convolutional layer
conv_depth_2 = 64       # Number of kernels in the second convolutional layer
drop_prob_1 = 0.25      # Probability of dropout after pooling
drop_prob_2 = 0.5       # Probability of dropout in the fully-connected layer
hidden_size = 512       # Number of nerons in the fully-connected layer

"""
1. Loading and pre-processing the CIFAR-10 data set
Do not initially consider each pixel as an independent inout feature 
    Do not reshape to 1D
    Force pixel intensities to the range [0, 1]
    Use one-hot encoding for the output labels
"""

(x_train, y_train), (x_test, y_test) = cifar10.load_data()      # Get CIFAR-10 data
num_train, height, width, depth = x_train.shape
num_test = x_test.shape[0]
num_classes = np.unique(y_train).shape[0]                       # Find the number of classes

x_train = x_train.astype("float32")                             # Convert to flaot32
x_test = x_test.astype("float32")
x_train /= np.max(x_train)                                      # Normalise
x_test /= np.max(x_test)

y_train = np_utils.to_categorical(y_train, num_classes)         # One-hot encode the labels
y_test = np_utils.to_categorical(y_test, num_classes)           # One-hot encode the labels

"""
2. Define the model
The model will consist of four Convolution2D layers
There will be a MaxPooling2D layer after the second and fourth convolutions
After the first pooling layer, the number of kernels is doubled
Afterwards, the output of the second pooling layer is flattened to 1D
Then passed through two Dense layers
ReLU activations will be used for all layers except the output dense layer
NB: Keras has an internal flag that automatically enables or disables dropout
    Depending on whether the model is currently used for training or testing
We will report accuracy as the data set is balance
We will hold out 10% of the data for validation purposes
"""

inp = Input(shape=(height, width, depth))

conv_1 = Convolution2D(filters=conv_depth_1,
                       kernel_size=(kernel_size, kernel_size),
                       padding="same",
                       activation="relu")(inp)

conv_2 = Convolution2D(filters=conv_depth_1,
                       kernel_size=(kernel_size, kernel_size),
                       padding="same",
                       activation="relu")(conv_1)

pool_1 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_2)

drop_1 = Dropout(rate=drop_prob_1)(pool_1)

conv_3 = Convolution2D(filters=conv_depth_2,
                       kernel_size=(kernel_size, kernel_size),
                       padding="same",
                       activation="relu")(drop_1)

conv_4 = Convolution2D(filters=conv_depth_2,
                       kernel_size=(kernel_size, kernel_size),
                       padding="same",
                       activation="relu")(conv_3)

pool_2 = MaxPooling2D(pool_size=(pool_size, pool_size))(conv_4)

drop_2 = Dropout(rate=drop_prob_1)(pool_2)

flat = Flatten()(drop_2)

hidden = Dense(units=hidden_size,
               activation="relu")(flat)

drop_3 = Dropout(rate=drop_prob_2)(hidden)

out = Dense(units=num_classes,
            activation="softmax")(drop_3)

model = Model(inputs=inp, outputs=out)

model.compile(loss="categorical_crossentropy",
              optimizer="adam",
              metrics=["accuracy"])

model.fit(x=x_train,
          y=y_train,
          epochs=num_epochs,
          verbose=1,
          validation_split=0.1)

scores = model.evaluate(x=x_test,
                        y=y_test,
                        verbose=1)

print("\n")
for name, score in zip(* (model.metrics_names, scores)):
    print(name + ": " + str(round(score, 5)))


