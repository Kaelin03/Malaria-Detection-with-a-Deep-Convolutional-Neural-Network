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
        self._image_size = None
        self._model = None
        self._configure(config)

    def train(self, x, y):
        # TODO: get num_classes from training set
        self._compile_model(image_size=self._image_size, num_classes=2)     # Build the CNN model
        y = np_utils.to_categorical(y=y, num_classes=2)                     # One-hot encode the labels
        self._model.fit(x=x, y=y,
                        epochs=self._num_epochs,
                        verbose=1,
                        validation_split=0.1)                               # Train the CNN

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

    def _configure(self, config):
        # Given the name of the classifier configuration file
        # Sets each hyper-parameter of the classifier
        print("Configuring cell detector...")
        config_path = "../config/" + config
        if os.path.isfile(config_path):
            with open(config_path) as file:
                config_dict = yaml.load(file)
            try:
                self._batch_size = config_dict["batch_size"]                        # Set batch size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: batch_size not found.")
            try:
                self._num_epochs = config_dict["num_epochs"]                        # Set number of epochs
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: num_epochs not found.")
            try:
                self._kernel_size = config_dict["kernel_size"]                      # Set kernel_size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: kernel_size not found.")
            try:
                self._pool_size = config_dict["pool_size"]                          # Set pool_size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: pool_size not found.")
            try:
                self._conv_depth = config_dict["conv_depth"]                        # Set conv depth
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: conv_depth not found.")
            try:
                self._drop_prob = config_dict["drop_prob"]                          # Set drop probability
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: drop_prob not found.")
            try:
                self._hidden_size = config_dict["hidden_size"]                      # Set hidden size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: hidden_size not found.")
            try:
                self._image_size = (config_dict["image_width"],
                                    config_dict["image_height"],
                                    config_dict["image_width"])                     # Set image size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: image sizes not found.")
            print("Done.")
        else:
            print("Warning: " + config + " not found.")

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
