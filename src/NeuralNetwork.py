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
import matplotlib.pyplot as plt
import numpy as np
import yaml
import os


class NeuralNetwork(object):

    def __init__(self, yaml_path):
        """
        Initialises all Class attributes and updates them with values given in the config file
        :param yaml_path: path to the configuration file
        """
        self._yaml_path = "../config/" + yaml_path
        self._name = ""
        self._kernel_size = None
        self._batch_size = None
        self._augment = None
        self._drop_prob = None
        self._conv_depth = None
        self._hidden_size = None
        self._learn_rate = None
        self._epochs = None
        self._pool_size = None
        self._activation = None
        self._model = None
        self._history = None
        self._train_size = None
        self.update_config()

    def is_ready(self):
        if self._model is None:
            return False
        else:
            return True

    def get_name(self):
        """
        :return:
        """
        return self._name

    def update_config(self):
        """
        Updates all attributes as per the given configuration file
        :return: None
        """
        try:
            config = self._get_yaml_dict(self._yaml_path)["neural_network"]
            self._kernel_size = [(value, value) for value in config["kernel_size"]]
            self._batch_size = config["batch_size"]
            self._augment = config["augment"]
            self._drop_prob = config["drop_prob"]
            self._conv_depth = config["conv_depth"]
            self._hidden_size = config["hidden_size"]
            self._learn_rate = config["learn_rate"]
            self._epochs = config["epochs"]
            self._activation = config["activation"]
            self._pool_size = [(value, value) for value in config["pool_size"]]
        except KeyError:
            print("Warning: Some entries were missing from the neural_network configurations.")
            
    def train(self, x, y):
        """
        :param x: training images
        :param y: corresponding labels
        :return: None
        """
        self.update_config()
        num_classes = np.unique(y).shape[0]                                         # Find the number of classes
        image_shape = x.shape[1:4]                                                  # Get the image shape
        self.compile_model(image_shape, num_classes)                                # Compile the model
        self._train_size = [y[np.where(y == label)].shape[0] for label in range(num_classes)]
        y = np_utils.to_categorical(y=y, num_classes=num_classes)                   # One-hot encode the labels
        if self._augment:
            print("Warning: augment option not available yet.")
        else:
            self._history = self._model.fit(x=x, y=y,
                                            batch_size=self._batch_size,
                                            epochs=self._epochs,
                                            verbose=1,
                                            validation_split=0.1)                       # Train the CNN
        while True:
            self._name = input("Please enter a name for the model:\n")
            if self._name != "":
                break
        self.save_all()

    def save_all(self):
        """
        :return:
        """
        while True:
            option = input("Would you like to save the model? y/n\n")
            if option == "y" or option == "n":
                break
        if option == "y":
            self.save_model()
        while True:
            option = input("Would you like to draw the model? y/n\n")
            if option == "y" or option == "n":
                break
        if option == "y":
            self.draw_model()
        while True:
            option = input("Would you like to save the model history? y/n\n")
            if option == "y" or option == "n":
                break
        if option == "y":
            self.save_history()
        while True:
            option = input("Would you like to plot the model history? y/n\n")
            if option == "y" or option == "n":
                break
        if option == "y":
            self.plot_history()

    def evaluate(self, x, y, num_classes=0):
        """
        If number of classes is zero, the number of classes will be deduced from the given labels
        :param x: test images
        :param y: corresponding labels
        :param num_classes: number of classes
        :return:
        """
        if self._model is not None:
            if not num_classes:
                num_classes = np.unique(y).shape[0]                                     # Find the number of classes
            y = np_utils.to_categorical(y=y, num_classes=num_classes)                   # One-hot encode the labels
            scores = self._model.evaluate(x=x,
                                          y=y,
                                          verbose=0)                                    # Evaluate using test data
            print("Class " + str(int(np.where(y[0] == 1)[0])) + ":")
            print("\tNumber of images:" + str(x.shape[0]))
            for name, score in zip(*(self._model.metrics_names, scores)):               # Print out scores
                print("\t" + name + ": " + str(round(score, 5)))
        else:
            print("Warning: model is not yet compiled.")

    def draw_model(self):
        """
        Saves a diagram of the model
        :return: None
        """
        if self._model is not None:
            directory = "../models/diagrams"                                    # Set destination directory
            self._make_path(directory)                                          # Make destination directory
            model_name = input("Enter the image name (default is model name):\n")
            if not model_name:
                model_name = self._name
            if model_name[-4] != ".jpg" or model_name[-4] != ".png":            # If extension not given
                model_name += ".png"                                            # Add extension
            plot_model(self._model,
                       to_file=directory + "/" + model_name,
                       show_shapes=True,
                       show_layer_names=False)                                  # Save a plot of the model
            print("Diagram saved to " + directory + "/" + model_name + ".")     # Feed back for user
        else:
            print("Warning: model not compiled yet.")                           # Feed back for user

    def plot_history(self):
        """
        Saves a plot of the acc and loss
        :return: None
        """
        if self._model is not None:
            directory = "../figures"
            self._make_path(directory)
            loss = self._history.history["loss"]
            val_loss = self._history.history["val_loss"]
            acc = self._history.history["acc"]
            val_acc = self._history.history["val_acc"]
            plt.figure(1)
            plt.subplot(121)
            plt.plot(acc, "red", label="Accuracy")
            plt.plot(val_acc, "blue", label="Validation accuracy")
            plt.legend(loc=0, frameon=False)
            plt.ylabel("Accuracy")
            plt.xlabel("Epoch")
            plt.grid()
            plt.subplot(122)
            plt.plot(loss, "red", label="Training loss")
            plt.plot(val_loss, "blue", label="Validation loss")
            plt.legend(loc=0, frameon=False)
            plt.ylabel("Loss")
            plt.xlabel("Epoch")
            plt.grid()
            while True:
                save = input("Would you like to save the plot? y/n\n")
                if save == "y" or save == "n":
                    break
            if save == "y":
                plot_name = input("Enter the figure name (default is model name):\n")
                if plot_name == "":
                    plot_name = self._name
                if plot_name[-4] != ".jpg" or plot_name[-4] != ".png":
                    plot_name += ".png"
                plt.savefig(directory + "/" + plot_name)
            plt.show()

    def save_history(self):
        """
        :return:
        """
        directory = "../logs"                                                   # Set destination directory
        self._make_path(directory)                                              # Make destination directory
        filename = input("Enter the image name (default is model name):\n")
        if not filename:
            filename = self._name
        if filename[-4] != ".csv" or filename[-4] != ".txt":                    # If extension not given
            filename += ".csv"                                                  # Add extension
        file = open(directory + "/" + filename, "w")                            # Open file
        file.write("epochs," + str(self._epochs) + "\n")
        file.write("kernel size," + self._format(self._kernel_size) + "\n")
        file.write("hidden size," + self._remove_brackets(self._hidden_size) + "\n")
        file.write("pool size," + self._format(self._pool_size) + "\n")
        file.write("conv depth," + self._remove_brackets(self._conv_depth) + "\n")
        file.write("drop prob," + self._remove_brackets(self._drop_prob) + "\n")
        file.write("learn rate," + str(self._learn_rate) + "\n")
        file.write("augment," + str(self._augment) + "\n")
        file.write("activation," + str(self._activation) + "\n")
        file.write("batch size," + str(self._batch_size) + "\n")
        file.write("train size," + self._remove_brackets(self._train_size) + "\n\n")
        keys = [key for key in self._history.history]                           # For each key in the history dict
        epochs = len(self._history.history[keys[0]])                            # Get the total number of epochs
        file.write(",".join(keys) + "\n")                                       # Make a comma sep. string from keys
        for i in range(epochs):
            data = [self._history.history[key][i] for key in keys]              # Get all data for a given epoch
            data = ",".join(list(map(str, data)))                               # Convert to comma separated string
            file.write(data + "\n")                                             # Write to file
        file.close()
        print("History saved to " + directory + "/" + filename + ".")           # Feed back for user

    def save_model(self):
        """
        Saves the model
        :return: None
        """
        if self._model is not None:
            directory = "../models"
            self._make_path(directory)
            model_name = self._name
            if model_name[-3] != ".h5":                                             # If extension not given
                model_name += ".h5"                                                 # Add extension
            self._model.save(filepath=directory + "/" + model_name)                 # Save model
            print("Model saved to " + directory + "/" + model_name + ".")
        else:
            print("Warning: model not compiled yet.")

    def load_model(self):
        """
        Loads a model from user input
        :return: None
        """
        model_name = input("Enter the model name:\n")
        directory = "../models"
        if model_name[-3] != ".h5":
            model_name += ".h5"
        if os.path.isfile(directory + "/" + model_name):
            self._model = load_model(directory + "/" + model_name)
            self._name = model_name[0:-3]
            print("Successfully loaded " + model_name + ".")
        else:
            print("Warning: " + directory + "/" + model_name + " not found.")

    def predict(self, x):
        """
        :param x: images
        :return: predicted labels for given images
        """
        predictions = self._model.predict(x, verbose=1)
        return predictions

    def compile_model(self, image_shape, num_classes):
        """
        :param image_shape: shape of input images
        :param num_classes: number of classes
        :return:
        """
        self._model = Sequential()
        # Conv2D, Conv2D, Pool
        self._model.add(Conv2D(self._conv_depth[0], self._kernel_size[0],
                               input_shape=image_shape,
                               padding="same",
                               activation=self._activation))
        self._model.add(Conv2D(self._conv_depth[1], self._kernel_size[1],
                               padding="same",
                               activation=self._activation))
        self._model.add(MaxPooling2D(self._pool_size[0]))

        # Dropout
        self._model.add(Dropout(self._drop_prob[0]))

        # Conv2D, Conv2D, Pool
        self._model.add(Conv2D(self._conv_depth[2], self._kernel_size[2],
                               padding="same",
                               activation=self._activation))
        self._model.add(Conv2D(self._conv_depth[3], self._kernel_size[3],
                               padding="same",
                               activation=self._activation))
        self._model.add(MaxPooling2D(self._pool_size[1]))

        # Dropout
        self._model.add(Dropout(self._drop_prob[1]))

        # Conv2D, Conv2D, Pool
        self._model.add(Conv2D(self._conv_depth[4], self._kernel_size[4],
                               padding="same",
                               activation=self._activation))
        self._model.add(Conv2D(self._conv_depth[5], self._kernel_size[5],
                               padding="same",
                               activation=self._activation))
        self._model.add(MaxPooling2D(self._pool_size[2]))

        # Dropout
        self._model.add(Dropout(self._drop_prob[2]))

        # Flatten
        self._model.add(Flatten())

        # Dense, Dropout, Dense
        self._model.add(Dense(self._hidden_size[0],
                        activation=self._activation))
        self._model.add(Dropout(self._drop_prob[3]))
        self._model.add(Dense(num_classes,
                        activation="softmax"))

        opt = rmsprop(lr=self._learn_rate, decay=1e-6)

        self._model.compile(loss="categorical_crossentropy",
                            optimizer=opt,
                            metrics=["accuracy"])

    @staticmethod
    def _remove_brackets(values):
        """
        :param values:
        :return:
        """
        values = str(values)
        values = values.replace("[", "")
        values = values.replace("]", "")
        values = values.replace("(", "")
        values = values.replace(")", "")
        values = values.replace(" ", "")
        return values

    @staticmethod
    def _format(values):
        """
        :param values:
        :return:
        """
        values = str([value[0] for value in values])
        values = values.replace("[", "")
        values = values.replace("]", "")
        values = values.replace(" ", "")
        return values

    @staticmethod
    def _make_path(path):
        """
        :param path:
        :return:
        """
        directories = path.split("/")                                           # Get list of directories
        path = ""                                                               # Init path string
        for directory in directories:                                           # For each directory
            path += directory + "/"                                             # Add it to the path
            if not os.path.isdir(path):                                         # If it does not yet exist
                os.mkdir(path)                                                  # Make it

    @staticmethod
    def _get_yaml_dict(yaml_path):
        """
        :param yaml_path: path to a yaml file (string)
        :return: contents of the yaml file (dict
        """
        yaml_dict = {}                                                              # Initialise yaml_dict
        if os.path.isfile(yaml_path):                                               # If the file exists
            with open(yaml_path) as file:                                           # Open the file
                yaml_dict = yaml.load(file)                                         # Load the file
        else:
            print("Warning: " + yaml_path + " not found.")
        return yaml_dict

