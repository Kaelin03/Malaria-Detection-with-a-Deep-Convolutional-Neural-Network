#!/usr/bin/env python3

from keras.models import Sequential                             # Class for specifying and training a Neural Network
from keras.models import load_model                             # For loading models
from keras.layers import Conv2D                                 # For convolutional layers
from keras.layers import MaxPooling2D                           # For max pooling layers
from keras.layers import Dense                                  # For fully-connected layers
from keras.layers import Dropout                                # For dropout layers
from keras.layers import Flatten                                # For converting to 1D
from keras.utils import plot_model                              # For graphical representation of model
from keras.utils import np_utils                                # Utilities for one-hot encoding of ground truth values
from keras.preprocessing.image import ImageDataGenerator        # For data augmentation
from keras.optimizers import rmsprop
import numpy as np
import os


class Classifier(object):

    def __init__(self, config):
        self._kernel = config["kernel_size"]                                        # Get kernel size
        self._batch_size = config["batch_size"]                                     # Get batch size
        self._epochs = config["num_epochs"]                                         # Get number of epochs
        self._pool = config["pool_size"]                                            # Get pool size
        self._conv_depth = tuple(config["conv_depth"])                              # Get conv depth (nb of filters)
        self._drop_prob = tuple(config["drop_prob"])                                # Get drop prob
        self._hidden_size = config["hidden_size"]                                   # Get hidden layer size
        self._augment = config["augment"]                                           # Real time data augmentation
        self._learn_rate = config["learn_rate"]
        self._model = None                                                          # Initialise model

    def train(self, x, y):
        num_classes = np.unique(y).shape[0]                                         # Find the number of classes
        image_size = x.shape[1:4]                                                   # Find the image size
        self.compile_model(image_size=image_size, num_classes=num_classes)          # Build the CNN model
        y = np_utils.to_categorical(y=y, num_classes=num_classes)                   # One-hot encode the labels
        if self._augment:
            datagen = ImageDataGenerator(
                featurewise_center=False,                                       # Set input mean to 0 over the dataset
                samplewise_center=False,                                        # Set each sample mean to 0
                featurewise_std_normalization=False,                            # Divide inputs by std of the dataset
                samplewise_std_normalization=False,                             # Divide each input by its std
                zca_whitening=False,                                            # Apply ZCA whitening
                rotation_range=0,                                               # Randomly rotate images
                width_shift_range=0.1,                                          # Randomly shift images horizontally
                height_shift_range=0.1,                                         # Randomly shift images vertically
                horizontal_flip=True,                                           # Randomly flip images
                vertical_flip=True)                                             # Randomly flip images
            datagen.fit(x)
            self._model.fit_generator(datagen.flow(x=x, y=y,
                                                   batch_size=self._batch_size),
                                      steps_per_epoch=x.shape[0] / self._batch_size,
                                      epochs=self._epochs,
                                      verbose=1,
                                      validation_data=(x, y))
        else:
            self._model.fit(x=x, y=y, batch_size=self._batch_size,
                            epochs=self._epochs,
                            verbose=1,
                            validation_split=0.1)                                       # Train the CNN
        while True:
            option = input("Would you like to save the model? y/n\n")
            if option == "y" or option == "n":
                break
        if option == "y":
            self.save_model()

    def save_model(self):
        if self._model is not None:
            model_name = input("Please enter the model name:\n")
            file_path = "../models/" + model_name + ".h5"
            self._model.save(filepath=file_path)                                        # Save model to file_path
            print("Model saved in models/" + model_name + ".h5")
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
        filename = input("Enter the model name:\n")
        filename = "../models/" + filename + ".png"
        if self._model is not None:
            plot_model(self._model,
                       to_file=filename,
                       show_shapes=True,
                       show_layer_names=False)
        else:
            print("Warning: model not compiled yet.")

    def evaluate(self, x, y, num_classes=0):
        if self._model is not None:
            if not num_classes:
                num_classes = np.unique(y).shape[0]                                     # Find the number of classes
            y = np_utils.to_categorical(y=y, num_classes=num_classes)                   # One-hot encode the labels
            scores = self._model.evaluate(x=x,
                                          y=y,
                                          verbose=0)                                    # Evaluate using test data
            print("Class " + str(int(np.where(y[0] == 1)[0])) + ":")
            for name, score in zip(*(self._model.metrics_names, scores)):               # Print out scores
                print(name + ": " + str(round(score, 5)))

    def test(self, x):
        predictions = self._model.predict(x)
        # TODO: Process and return predictions

    def compile_model(self, image_size, num_classes):
        self._model = Sequential()
        self._model.add(Conv2D(self._conv_depth[0], [self._kernel[0]] * 2,
                               input_shape=image_size,
                               padding="same",
                               activation="relu"))
        self._model.add(Conv2D(self._conv_depth[1], [self._kernel[1]] * 2,
                               padding="same",
                               activation="relu"))
        self._model.add(MaxPooling2D([self._pool[0]] * 2))
        self._model.add(Dropout(self._drop_prob[0]))
        self._model.add(Conv2D(self._conv_depth[2], [self._kernel[2]] * 2,
                               padding="same",
                               activation="relu"))
        self._model.add(Conv2D(self._conv_depth[3], [self._kernel[3]] * 2,
                               padding="same",
                               activation="relu"))
        self._model.add(MaxPooling2D([self._pool[1]] * 2))
        self._model.add(Dropout(self._drop_prob[1]))
        self._model.add(Flatten())
        self._model.add(Dense(self._hidden_size[0],
                        activation="relu"))
        self._model.add(Dropout(self._drop_prob[2]))
        self._model.add(Dense(num_classes,
                        activation="softmax"))
        opt = rmsprop(lr=self._learn_rate, decay=1e-6)

        self._model.compile(loss="categorical_crossentropy",
                            optimizer=opt,
                            metrics=["accuracy"])
