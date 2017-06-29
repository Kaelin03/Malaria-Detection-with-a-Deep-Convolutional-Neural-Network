#!/usr/bin/env python3

import os
import cv2
import yaml

from Image import Image
from CellDetector import CellDetector
from Classifier import Classifier


class DiagnosisSystem(object):

    def __init__(self, config):
        self._config = config
        self._train_cells = None                                            # Cell images for CNN training
        self._evaluation_cells = None                                       # Cell images for CNN evaluation
        self._samples = None                                                # Samples to manually classify
        self._test_samples = None                                           # Samples for testing
        self._cell_detector = CellDetector(config["cell_detector"])         # Initialise the cell detector
        self._classifier = Classifier(config["classifier"])                 # Initialise classifier

    def train(self):
        # TODO: Import cell images from train.yaml
        # TODO: Compile training labels
        # TODO: Call classifier.train(self._train_cells) to train the CNN
        pass

    def evaluate(self):
        # TODO: Load images of cells
        # TODO: Compile training labels
        # TODO: Call classifier.evaluate(x, y)
        pass

    def test(self):
        # TODO: Load samples
        # TODO: For each image
        # TODO: Detect cells
        # TODO: classify cells
        pass

    def manually_classify(self):
        # Creates train and test Image objects as per the manually_classify config file
        # Allows the user to manually classify each detected cell
        # Saves the cell into a directory with the same class name
        yaml_name = self._config["manually_classify"]                           # Get yaml file name
        yaml_path = "../config/" + yaml_name                                    # Get yaml file path
        if os.path.isfile(yaml_path):                                           # Check yaml file path is True
            with open(yaml_path) as yaml_file:                                  # Open yaml file
                yaml_dict = yaml.load(yaml_file)                                # Get dict from yaml file
            train_dirs = yaml_dict["train"]["directories"]                      # List of training dirs
            test_dirs = yaml_dict["test"]["directories"]                        # List of test dirs
            train_files = yaml_dict["train"]["files"]                           # List of training images
            test_files = yaml_dict["test"]["files"]                             # List of test images
            train_images = self._load_images(train_dirs, train_files)           # Load training image paths
            test_images = self._load_images(test_dirs, test_files)              # Load test image paths
            for image in train_images:                                          # For each training image
                cells = self._cell_detector.run(image)                          # Detect cells
                for cell in cells:                                              # For each cell
                    image.add_cell(cell)                                        # Add cell object to image
            for image in test_images:                                           # For each test image
                cells = self._cell_detector.run(image)                          # Detect cells
                for cell in cells:                                              # For each cell
                    image.add_cells(cell)                                       # Add cell to object image
            # TODO: Show cells for manual classification
        else:
            print("Warning: " + yaml_name + " not found.")

    def _load_images(self, directories, files):
        # Given a list of directories and a list of image file paths
        # Gets all valid image paths from the directories
        # Gets all valid image paths from the list of files
        # Returns a list of Image objects
        image_paths = []                                                        # List to store image paths
        image_paths += self._get_image_paths(directories)                       # Load image paths from directories
        image_paths += self._check_images(files)                                # Load image paths from file paths
        image_paths = set(image_paths)                                          # Remove any duplicates
        images = [Image(path) for path in image_paths]                          # Create an image object for each path
        return images                                                           # Return image paths

    @staticmethod
    def _check_images(images):
        # Given a list of file paths
        # Removes any duplicates or non png/jpg files
        # Returns the list of file paths
        image_paths = []                                                        # List to store image paths
        if isinstance(images, list):                                            # Check a list has been given
            for image in images:                                                # For each image_path
                image = "../" + image                                           # The path will begin up one level
                if os.path.isfile(image):                                       # Check the given path is a file
                    ext = image.split(".")[-1]                                  # Get the file extension
                    if ext == "jpg" or "png":                                   # If the image is a png or jpg
                        image_paths.append(image)                               # Append image path to image_paths
        return image_paths                                                      # Return image paths

    @staticmethod
    def _get_image_paths(directories):
        # Given a list of directories
        # Returns a list of the image paths within the directory
        image_paths = []                                                        # List to store image paths
        if isinstance(directories, list):                                       # If a list is given
            for directory in directories:                                       # For each directory
                directory = "../" + directory                                   # Directory will be one level above
                if os.path.isdir(directory):                                    # If directory is True
                    for file in os.listdir(directory):                          # For each file in the dir
                        file = directory + "/" + file                           # Get full file path
                        if os.path.isfile(file):                                # If the file exists
                            ext = file.split(".")[-1]                           # Get the file extension
                            if ext == "jpg" or ext == "png":                    # If file extension is jpg or png
                                image_paths.append(file)                        # Append the file to image_paths
        return image_paths                                                      # Return image paths

    @staticmethod
    def _get_sample_index(samples, sample_id):
        # Given a sample name
        # Returns the index of the sample if it exists
        # Returns -1 if the sample does not exist
        index = -1                                                              # Set index to -1
        for i, sample in enumerate(samples):                                    # For each sample
            if sample_id == sample.get_id():                                    # If the sample ids match
                index = i                                                       # Store the index
                break                                                           # Break
        return index                                                            # Return the index

    @staticmethod
    def _get_sample_id(image_path):
        # Given the path to an image
        # Returns the image name from the string
        # Returns False if no sample id is found
        try:
            sample_id = image_path.split("/")[-1].split("_")[-2]                # Extract the sample id
        except IndexError:
            print("Warning: no image name found in " + image_path + ".")
            sample_id = False
        return sample_id

