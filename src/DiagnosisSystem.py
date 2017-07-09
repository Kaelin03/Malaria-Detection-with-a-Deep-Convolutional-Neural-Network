#!/usr/bin/env python3

import os
import cv2
import yaml
import random
import numpy as np

from CellDetector import CellDetector
from NeuralNetwork import NeuralNetwork
from ManualClassifier import ManualClassifier


class DiagnosisSystem(object):

    def __init__(self, yaml_path):
        self._yaml_path = "../config/" + yaml_path
        self._manual_classifier = ManualClassifier(yaml_path)
        self._neural_network = NeuralNetwork(yaml_path)
        self._image_shape = None
        self._resize = None
        self._train_config = None
        self._evaluate_config = None
        self._diagnose_config = None
        self.update_config()

    def update_config(self):
        try:
            self._train_config = self._get_yaml_dict(self._yaml_path)["train"]
            self._evaluate_config = self._get_yaml_dict(self._yaml_path)["evaluate"]
            self._diagnose_config = self._get_yaml_dict(self._yaml_path)["diagnose"]
            image_config = self._get_yaml_dict(self._yaml_path)["images"]
            self._image_shape = (image_config["width"],
                                 image_config["height"],
                                 image_config["depth"])
            self._resize = image_config["resize"]
        except KeyError:
            print("Warning: Some entries were missing from the configuration file.")

    def manually_classify(self):
        self._manual_classifier.run()

    def train(self):
        self.update_config()
        data = []
        for label in self._train_config["labels"]:
            directories = self._train_config["labels"][label]["directories"]
            if directories is not None:
                data += self._get_cell_data(directories, label)
        if data:
            if self._train_config["small_images"] == "ignore":
                data = self._ignore_small(data, self._image_shape)              # Remove small images
            elif self._train_config["small_images"] == "pad":
                data = self._pad_with(data, self._image_shape, 255)             # Pad small images with zeros
            else:
                print("No changes made to image shapes.")
            random.shuffle(data)                                                # Shuffle the data
            x, y = self._to_arrays(data, self._image_shape)                     # Put the data into numpy arrays
            x = self._resize_images(x, self._resize)                            # Resize images
            x = self._normalise(x)
            cont = self._check_balance(y)                                       # Check that data is balanced
            if cont:
                self._neural_network.train(x, y)

    def evaluate(self):
        self.update_config()
        num_classes = len(self._evaluate_config["labels"])                              # Get number of classes
        for label in self._evaluate_config["labels"]:                                   # Evaluate each label separately
            directories = self._evaluate_config["labels"][label]["directories"]         # Get directories
            if directories is not None:                                                 # If directories given
                data = self._get_cell_data(directories, label)                          # Get imgs from list of dirs
                if data:                                                                # If there are images
                    if self._evaluate_config["small_images"] == "ignore":
                        data = self._ignore_small(data, self._image_shape)              # Remove small images
                    elif self._evaluate_config["small_images"] == "pad":
                        data = self._pad_with(data, self._image_shape, 255)             # Pad small images with zeros
                    else:
                        print("No changes made to image shapes.")
                    x, y = self._to_arrays(data, self._image_shape)
                    x = self._resize_images(x, self._resize)                            # Resize images
                    x = self._normalise(x)
                    self._neural_network.evaluate(x, y, num_classes=num_classes)

    def load_model(self):
        self._neural_network.load_model()

    def save_model(self):
        self._neural_network.save_model()

    def plot_model(self):
        self._neural_network.plot_model()

    def plot_history(self):
        self._neural_network.plot_history()

    def save_history(self):
        self._neural_network.save_history()

    def plot_filters(self):
        self._neural_network.plot_filters()

    def diagnose(self):
        self.update_config()
        directories = self._diagnose_config["directories"]
        if directories is not None:
            iamges = []
            # TODO: Create sample objects,
            # TODO: Convert cell images to np array

    def _get_cell_data(self, directories, label):
        data = []
        directories = ["../" + directory for directory in directories]              # Change initial dir
        for directory in directories:                                               # For each directory
            if os.path.isdir(directory):                                            # If the directory exists
                image_paths = os.listdir(directory)                                 # Get the file names in the dir
                image_paths = [directory + "/" + path for path in image_paths]      # Get the full file path
                image_paths = self._check_images(image_paths)                       # Check the files are images
                for path in image_paths:
                    data.append([cv2.imread(path), label])                          # Append image and label to data
            else:
                print("Warning: " + directory + " not found.")
        return data

    @staticmethod
    def _resize_images(x, size):
        if size is not None:
            x_resized = np.empty((x.shape[0], size, size, x.shape[3]), dtype=np.uint8)
            for i in range(x.shape[0]):
                x_resized[i] = cv2.resize(x[i], (size, size))
            x = x_resized
        return x

    @staticmethod
    def _normalise(x):
        x = x.astype("float32")                                                     # Convert to float32
        x /= 255                                                                    # Normalise
        return x

    @staticmethod
    def _check_balance(y):
        classes = np.unique(y)  # Find the number of classes
        class_freq = []
        cont = True
        for label in classes:
            class_freq.append((np.sum(y == label)))
        if not all(class_freq[0] == item for item in class_freq):
            print("Warning: unbalanced data, " + str(class_freq) + ".")
            while True:
                cont = input("Continue ? y/n\n")
                if cont == "y":
                    break
                elif cont == "n":
                    cont = False
                    break
        return cont

    @staticmethod
    def _to_arrays(data, shape):
        y = np.empty((len(data), 1), dtype=np.uint8)                                # Init labels array
        x = np.empty((len(data), shape[0], shape[1], shape[2]), dtype=np.uint8)     # Init data array
        for i, item in enumerate(data):                                             # Put data into numpy arrays
            x[i] = item[0]                                                          # x contains images
            y[i] = item[1]                                                          # y contains labels
        return x, y

    @staticmethod
    def _pad_with(data, shape, value=0):
        padded_data = []
        for item in data:
            if item[0].shape != shape:
                x, y, z = item[0].shape
                dx = int((shape[0] - x) / 2)
                dy = int((shape[1] - y) / 2)
                tmp = np.zeros(shape)
                tmp.fill(value)
                tmp[dx:x + dx, dy:y + dy, 0:z] = item[0]
                item[0] = tmp
            padded_data.append(item)
        return padded_data

    @staticmethod
    def _ignore_small(data, shape):
        return [item for item in data if item[0].shape == shape]

    @staticmethod
    def _check_images(image_paths):
        """
        Checks that file paths are valid image and removes non jpg/png files
        :param image_paths: List of paths to images
        :return: List of paths to images
        """
        images = []                                                                 # List to store image paths
        for path in image_paths:                                                    # For each image_path
            if os.path.isfile(path):                                                # Check the given path is a file
                ext = path.split(".")[-1]                                           # Get the file extension
                if ext == "jpg" or "png":                                           # If the image is a png or jpg
                    images.append(path)                                             # Append image path to image_paths
        return images

    @staticmethod
    def _get_yaml_dict(yaml_path):
        yaml_dict = {}                                                              # Initialise yaml_dict
        if os.path.isfile(yaml_path):                                               # If the file exists
            with open(yaml_path) as file:                                           # Open the file
                yaml_dict = yaml.load(file)                                         # Load the file
        else:
            print("Warning: " + yaml_path + " not found.")
        return yaml_dict

