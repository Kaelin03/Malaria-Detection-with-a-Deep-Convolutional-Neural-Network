#!/usr/bin/env python3

from keras.models import Model          # Class for specifying and training a Neural Network
from keras.models import load_model
from keras.layers import Input
from keras.layers import Convolution2D
from keras.layers import MaxPooling2D
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.utils import plot_model
from keras.utils import np_utils        # Utilities for one-hot encoding of ground truth values
import numpy as np
import os


class Classifier(object):

    def __init__(self, config):
        self._kernel_size = config["kernel_size"]                                   # Get kernel size
        self._batch_size = config["batch_size"]                                     # Get batch size
        self._num_epochs = config["num_epochs"]                                     # Get number of epochs
        self._pool_size = config["pool_size"]                                       # Get pool size
        self._conv_depth = tuple(config["conv_depth"])                              # Get conv depth (nb of filters)
        self._drop_prob = tuple(config["drop_prob"])                                # Get drop prob
        self._hidden_size = config["hidden_size"]                                   # Get hidden layer size
        self._model = None                                                          # Initialise model

    def train(self, x, y):
        num_classes = np.unique(y).shape[0]                                         # Find the number of classes
        image_size = x.shape[1:4]                                                   # Find the image size
        self._compile_model(image_size=image_size, num_classes=num_classes)         # Build the CNN model
        y = np_utils.to_categorical(y=y, num_classes=num_classes)                   # One-hot encode the labels
        x = self._normalise(x)                                                      # Normalise images
        self._model.fit(x=x, y=y,
                        epochs=self._num_epochs,
                        verbose=1,
                        validation_split=0.1,
                        shuffle=True)                                               # Train the CNN
        while True:
            option = input("Would you like to save the model? y/n\n")
            if option == "y" or option == "n":
                break
        if option == "y":
            self.save_model()

    @staticmethod
    def _normalise(x):
        x = x.astype("float32")                                                     # Convert to float32
        x /= 255                                                                    # Normalise
        return x

    def save_model(self):
        if self._model is not None:
            model_name = input("Please enter the model name:\n")
            file_path = "../models/" + model_name + ".h5"
            self._model.save(filepath=file_path)                                        # Save model to file_path
            print("Model saved in models/" + model_name + ".")
        else:
            print("Warning: model not compiled yet.")

    def load_model(self):
        filename = input("Enter the model name:\n")
        filename = "../models/" + filename
        if os.path.isfile(filename):
            self._model = load_model(filename)
        else:
            print("Warning: " + filename + " not found.")

    def plot_model(self):
        filename = input("Enter the model name:")
        filename = "../models/" + filename
        if self._model is not None:
            plot_model(self._model,
                       to_file=filename,
                       show_shapes=True,
                       show_layer_names=False)
        else:
            print("Warning: model not compiled yet.")

    def evaluate(self, x, y):
        y = np_utils.to_categorical(y=y, num_classes=2)                             # One-hot encode the labels
        x, y, = self._normalise(x, y)
        scores = self._model.evaluate(x=x,
                                      y=y,
                                      verbose=1)                                    # Evaluate using test data
        print("\n")
        for name, score in zip(*(self._model.metrics_names, scores)):               # Print out scores
            print(name + ": " + str(round(score, 5)))

    def test(self, x):
        predictions = self._model.predict(x)
        # TODO: Process and return predictions

    def _compile_model(self, image_size, num_classes):
        # Convolutional Neural Network using Keras
        # All activations functions are ReLU
        # Input -> Convolutional2D -> Convolutional2D -> MaxPooling2D ->
        # Dropout ->Convolution2D -> Convolution2D -> Dropout ->
        # MaxPooling2D -> Flatten -> Dense -> Dropout -> Dense
        inp = Input(shape=image_size)
        conv_1 = Convolution2D(filters=self._conv_depth[0],
                               kernel_size=(self._kernel_size, self._kernel_size),
                               padding="same",
                               activation="relu")(inp)
        conv_2 = Convolution2D(filters=self._conv_depth[0],
                               kernel_size=(self._kernel_size, self._kernel_size),
                               padding="same",
                               activation="relu")(conv_1)
        pool_1 = MaxPooling2D(pool_size=(self._pool_size, self._pool_size))(conv_2)
        drop_1 = Dropout(rate=self._drop_prob[0])(pool_1)
        conv_3 = Convolution2D(filters=self._conv_depth[1],
                               kernel_size=(self._kernel_size, self._kernel_size),
                               padding="same",
                               activation="relu")(drop_1)
        conv_4 = Convolution2D(filters=self._conv_depth[1],
                               kernel_size=(self._kernel_size, self._kernel_size),
                               padding="same",
                               activation="relu")(conv_3)
        pool_2 = MaxPooling2D(pool_size=(self._pool_size, self._pool_size))(conv_4)
        drop_2 = Dropout(rate=self._drop_prob[0])(pool_2)
        flat = Flatten()(drop_2)
        hidden = Dense(units=self._hidden_size,
                       activation="relu")(flat)
        drop_3 = Dropout(rate=self._drop_prob[1])(hidden)
        out = Dense(units=num_classes,
                    activation="softmax")(drop_3)
        self._model = Model(inputs=inp, outputs=out)
        self._model.compile(loss="categorical_crossentropy",
                            optimizer="adam",
                            metrics=["accuracy"])
