#!/usr/bin/env python3

import os
import cv2
import yaml
import random
import numpy as np

from CellDetector import CellDetector
from NeuralNetwork import NeuralNetwork
from ManualClassifier import ManualClassifier
from Sample import Sample


class DiagnosisSystem(object):

    def __init__(self, yaml_path):
        """
        :param yaml_path:
        """
        self._yaml_path = "../config/" + yaml_path
        self._manual_classifier = ManualClassifier(yaml_path)
        self._neural_network = NeuralNetwork(yaml_path)
        self._cell_detector = CellDetector(yaml_path)
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
            directories = self._train_config["labels"][label]["directories"]    # Get directories from config
            if directories is not None:                                         # If directories are given
                data += self._get_cell_data(directories, label)                 # Make a list of images and labels
        if data:
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
                    x, y = self._to_arrays(data, self._image_shape)
                    x = self._resize_images(x, self._resize)                            # Resize images
                    x = self._normalise(x)
                    self._neural_network.evaluate(x, y, num_classes=num_classes)

    def load_model(self):
        """
        :return:
        """
        self._neural_network.load_model()

    def save_model(self):
        """
        :return:
        """
        self._neural_network.save_model()

    def draw_model(self):
        """
        :return:
        """
        self._neural_network.draw_model()

    def plot_history(self):
        """
        :return:
        """
        self._neural_network.plot_history()

    def save_history(self):
        """
        :return:
        """
        self._neural_network.save_history()

    def plot_filters(self):
        """
        :return:
        """
        self._neural_network.plot_filters()

    def diagnose(self):
        """
        :return:
        """
        self.update_config()
        directories = self._diagnose_config["directories"]
        samples = self._load_samples(directories)
        for sample in samples:                                                          # For each sample
            for image in sample.get_images():                                           # Gor each image
                cells = self._cell_detector.run(image)                                  # Detect cells
                image.add_cells(cells)                                                  # Add cells to image
                x = np.empty((len(cells),
                              self._image_shape[0],
                              self._image_shape[1],
                              self._image_shape[2]), dtype=np.uint8)
                for i, cell in enumerate(cells):
                    cell_image = cell.get_image(dx=self._image_shape[0],
                                                dy=self._image_shape[1])                # Put image into list
                    width, height, depth = cell_image.shape
                    x[i, 0:width, 0:height, 0:depth] = cell_image
                x = self._resize_images(x, self._resize)                                # Resize images
                x = self._normalise(x)
                predictions = self._neural_network.predict(x)
                img = image.get_image()
                for i, cell in enumerate(cells):
                    cell.set_status(np.argmax(predictions[i]))
                    cell.draw(img)
                self._save_image(img,
                                 image.get_name() + "." + image.get_type(),
                                 "../" + self._diagnose_config["destination"] + "/" + sample.get_id())
            print("Sample id: " + sample.get_id())
            print("\tTotal cells: " + str(sample.total_cells()))
            print("\tHealthy cells: " + str(sample.total_cells(0)))
            print("\tInfected cells: " + str(sample.total_cells(1)))

    def _load_samples(self, directories):
        """
        :param directories:
        :return:
        """
        print("Loading samples...")
        samples = []
        directories = ["../" + directory for directory in directories]                  # Change initial dir
        if directories is not None:
            for directory in directories:
                if os.path.isdir(directory):
                    image_paths = os.listdir(directory)
                    image_paths = [directory + "/" + path for path in image_paths]
                    image_paths = self._check_images(image_paths)
                    for image_path in image_paths:
                        sample_id = self._get_sample_id(image_path)
                        if sample_id is not None:
                            sample_index = self._get_sample_index(samples, sample_id)
                            if sample_index != -1:
                                samples[sample_index].add_image(image_path)
                            else:
                                sample = Sample(sample_id)
                                sample.add_image(image_path)
                                samples.append(sample)
                    if len(samples) == 1:
                        print("1 sample found.")
                    else:
                        print(str(len(samples)) + " samples found.")
                else:
                    print("Warning: " + directory + " not found.")
        else:
            print("Warning: No directories given.")
        return samples

    def _get_cell_data(self, directories, label):
        """
        :param directories:
        :param label:
        :return:
        """
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

    def _ignore_small(self, data):
        """
        :param data:
        :return:
        """
        return [item for item in data if item[0].shape == self._image_shape]

    @staticmethod
    def _pad_with(data, shape, value=0):
        """
        :param data:
        :param shape:
        :param value:
        :return:
        """
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
    def _resize_images(x, size):
        """
        :param x:
        :param size:
        :return:
        """
        if size is not None:
            x_resized = np.empty((x.shape[0], size, size, x.shape[3]), dtype=np.uint8)
            for i in range(x.shape[0]):
                x_resized[i] = cv2.resize(x[i], (size, size))
            x = x_resized
        return x

    @staticmethod
    def _normalise(x):
        """
        :param x:
        :return:
        """
        x = x.astype("float32")                                                     # Convert to float32
        x /= 255                                                                    # Normalise
        return x

    @staticmethod
    def _check_balance(y):
        """
        :param y:
        :return:
        """
        classes = np.unique(y)                                                      # Find the number of classes
        class_freq = []
        cont = True
        for label in classes:
            class_freq.append((np.sum(y == label)))
        if not all(class_freq[0] == item for item in class_freq):
            print("Warning: unbalanced data, " + str(class_freq) + ".")
            while True:
                cont = input("Continue? y/n\n")
                if cont == "y":
                    break
                elif cont == "n":
                    cont = False
                    break
        return cont

    @staticmethod
    def _get_sample_index(samples, sample_id):
        """
        :param samples:
        :param sample_id:
        :return:
        """
        index = -1
        for i, sample in enumerate(samples):
            if sample_id == sample.get_id():
                index = i
                break
        return index

    @staticmethod
    def _to_arrays(data, shape):
        """
        :param data:
        :param shape:
        :return:
        """
        y = np.empty((len(data), 1), dtype=np.uint8)                                # Init labels array
        x = np.zeros((len(data), shape[0], shape[1], shape[2]), dtype=np.uint8)     # Init data array
        for i, (image, label) in enumerate(data):                                   # Put data into numpy arrays
            height, width, depth = image.shape
            x[i, 0:height, 0:width, 0:depth] = image                                                            # x contains images
            y[i] = label                                                            # y contains labels
        return x, y

    @staticmethod
    def _get_sample_id(image_path):
        """
        :param image_path:
        :return:
        """
        try:
            sample_id = image_path.split("/")[-1].split("_")[-2]
        except IndexError:
            print("Warning: no image name found in " + image_path + ".")
            sample_id = None
        return sample_id

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
    def _save_image(image, image_name, destination):
        """
        Saves the image in a given directory
        creates the directory if it does not already exist
        :param image: image to save
        :param image_name: name to save image as
        :param destination: directory to save image in
        :return:
        """
        directories = destination.split("/")                                        # Get list of directories
        path = ""                                                                   # Init path string
        for directory in directories:                                               # For each directory
            path += directory + "/"                                                 # Add it to the path
            if not os.path.isdir(path):                                             # If it does not yet exist
                os.mkdir(path)                                                      # Make it
        cv2.imwrite(path + "/" + image_name, image)                                 # Save image to path

    @staticmethod
    def _get_yaml_dict(yaml_path):
        """
        :param yaml_path:
        :return:
        """
        yaml_dict = {}                                                              # Initialise yaml_dict
        if os.path.isfile(yaml_path):                                               # If the file exists
            with open(yaml_path) as file:                                           # Open the file
                yaml_dict = yaml.load(file)                                         # Load the file
        else:
            print("Warning: " + yaml_path + " not found.")
        return yaml_dict

