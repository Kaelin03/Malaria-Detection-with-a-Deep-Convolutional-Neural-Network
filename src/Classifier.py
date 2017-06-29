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
import yaml
import os


class Classifier(object):

    def __init__(self, config):
        self._batch_size = None
        self._num_epochs = None
        self._pool_size = None
        self._conv_depth = None
        self._drop_prob = None
        self._hidden_size = None
        self._model = None
        self._configure(config)

    def _configure(self, config):
        # Given the name of the classifier configuration file
        # Sets each hyper-parameter of the classifier
        print("Configuring cell detector...")
        config_path = "../config/" + config
        if os.path.isfile(config_path):
            with open(config_path) as file:
                config_dict = yaml.load(file)
            try:
                self._set_batch_size(config_dict["batch_size"][0])                  # Set batch size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: batch_size not found.")
            try:
                self._set_num_epochs(config_dict["num_epochs"][0])                  # Set number of epochs
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: num_epochs not found.")
            try:
                self._set_kernel_size(config_dict["kernel_size"][0])                # Set kernel_size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: kernel_size not found.")
            try:
                self._set_pool_size(config_dict["pool_size"][0])                    # Set pool_size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: pool_size not found.")
            try:
                self._set_conv_depth(config_dict["conv_depth"])                     # Set conv depth
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: conv_depth not found.")
            try:
                self._set_drop_prob(config_dict["drop_prob"])                       # Set drop probability
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: drop_prob not found.")
            try:
                self._set_hidden_size(config_dict["hidden_size"][0])                  # Set hidden size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: hidden_size not found.")
            print("Done.")
        else:
            print("Warning: " + config + " not found.")

    def _set_batch_size(self, batch_size):
        # Given the batch size
        # Sets the batch size
        try:
            self._batch_size = int(batch_size)                  # Convert to int
        except ValueError:
            self._batch_size = None                             # Signify batch_size not set
            print("Warning: invalid batch_size given.")

    def _set_num_epochs(self, num_epochs):
        # Given the number of epochs
        # Sets num_epochs
        try:
            self._num_epochs = int(num_epochs)                  # Convert to int
        except ValueError:
            self._num_epochs = None                             # Signify num_epochs not set
            print("Warning: invalid num_epochs given.")

    def _set_kernel_size(self, kernel_size):
        # Given the kernel size
        # Sets kernel_size
        try:
            self._kernel_size = int(kernel_size)                # Convert to int
        except ValueError:
            self._kernel_size = None                            # Signify kernel_size not set
            print("Warning: invalid kernel_size given.")

    def _set_pool_size(self, pool_size):
        # Given the pool size
        # Sets pool_size
        try:
            self._pool_size = int(pool_size)                     # Convert to int
        except ValueError:
            self._pool_size = None                              # Signify pool_size not set
            print("Warning: invalid pool_size given.")

    def _set_conv_depth(self, conv_depth):
        # Given the convolution layer depths
        # Sets conv_depth
        try:
            self._conv_depth = tuple(map(int, (conv_depth[0], conv_depth[1])))
        except ValueError:
            self._conv_depth = None                             # Signify batch_size not set
            print("Warning: invalid conv_depth given.")

    def _set_drop_prob(self, drop_prob):
        # Given the convolution layer depths
        # Sets drop_prob
        try:
            self._drop_prob = tuple(map(int, (drop_prob[0], drop_prob[1])))
        except ValueError:
            self._drop_prob = None                             # Signify batch_size not set
            print("Warning: invalid conv_depth given.")

    def _set_hidden_size(self, hidden_size):
        # Given the hidden size
        # Sets hidden_size
        try:
            self._hidden_size = int(hidden_size)                # Convert to int
        except ValueError:
            self._hidden_size = None                            # Signify batch_size not set
            print("Warning: invalid hidden_size given.")

    def _compile_model(self, image_size, num_classes):
        """
        Convolutional Neural Network using Keras
        All activations functions are ReLU
        Input -> Convolutional2D -> Convolutional2D -> MaxPooling2D ->
        Dropout ->Convolution2D -> Convolution2D -> Dropout ->
        MaxPooling2D -> Flatten -> Dense -> Dropout -> Dense
        :return: Nothing
        """
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
        model = Model(inputs=inp, outputs=out)
        model.compile(loss="categorical_crossentropy",
                      optimizer="adam",
                      metrics=["accuracy"])

    def train(self, x, y):
        # TODO: extract image_size and num_classes from training set
        self._compile_model(image_size=(64, 64, 3), num_classes=2)      # Build the CNN model
        y = np_utils.to_categorical(y=y, num_classes=2)                 # One-hot encode the labels
        self._model.fit(x=x, y=y,
                        epochs=self._num_epochs,
                        verbose=1,
                        validation_split=0.1)                           # Train the CNN

    def save_model(self, model_name):
        file_path = "../models/" + model_name + ".h5"
        self._model.save(filepath=file_path)                            # Save model to file_path
        print("Model saved in models/" + model_name + ".")

    def evaluate(self, x, y):
        y = np_utils.to_categorical(y=y, num_classes=2)                 # One-hot encode the labels
        scores = self._model.evaluate(x=x,
                                      y=y,
                                      verbose=1)                        # Evaluate using test data
        print("\n")
        for name, score in zip(*(self._model.metrics_names, scores)):   # Print out scores
            print(name + ": " + str(round(score, 5)))

    def test(self, x):
        predictions = self._model.predict(x)
        # TODO: Process and return predictions
