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
        self._train_cells = None                                                # Cell images for CNN training
        self._evaluation_cells = None                                           # Cell images for CNN evaluation
        self._samples = None                                                    # Samples to manually classify
        self._test_samples = None                                               # Samples for testing
        self._cell_detector = CellDetector(config["cell_detector"])             # Initialise the cell detector
        self._classifier = Classifier(config["classifier"])                     # Initialise classifier

    def train(self):
        yaml_name = self._config["train"]
        yaml_path = "../config/" + yaml_name
        yaml_dict = self._get_yaml_dict(yaml_path)
        if yaml_dict:
            for label in yaml_dict:
                train_dirs = yaml_dict[label]["directories"]                    # List of train dirs
        else:
            print("Warning: " + yaml_name + " not found.")
        # TODO: Import cell images from train.yaml
        # TODO: Compile training labels
        # TODO: Call classifier.train(self._train_cells) to train the CNN

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
        yaml_dict = self._get_yaml_dict(yaml_path)
        if yaml_dict:
            train_dirs = yaml_dict["train"]["directories"]                      # List of training dirs
            test_dirs = yaml_dict["test"]["directories"]                        # List of test dirs
            train_files = yaml_dict["train"]["files"]                           # List of training images
            test_files = yaml_dict["test"]["files"]                             # List of test images
            image_height = yaml_dict["image_height"]                            # Height to crop cell images to
            image_width = yaml_dict["image_width"]                              # Width to crop cell images to
            train_destination = yaml_dict["train"]["destination"]               # Get destination of training cells
            test_destination = yaml_dict["test"]["destination"]                 # Get destination of test cells
            labels = yaml_dict["labels"]                                        # Get names of classifications
            train_images = self._load_images(train_dirs, train_files)           # Load training image paths
            test_images = self._load_images(test_dirs, test_files)              # Load test image paths
            self._detect_and_show(train_images, train_destination, labels,
                                  image_width, image_height)                    # Detect cells and show user
            self._detect_and_show(test_images, test_destination, labels,
                                  image_width, image_height)                    # Detect cells and show user
        else:
            print("Warning: " + yaml_name + " not found.")

    def _detect_and_show(self, images, destination, labels, image_width=0, image_height=0):
        destination = "../" + destination                                       # Destination path starts one level up
        self._make_cell_dirs(destination, labels)
        for image in images:                                                    # For each image
            cells = self._cell_detector.run(image)                              # Detect cells
            image.add_cells(cells)                                              # Add cells to image
            img = image.get_image()                                             # Get image
            image.draw_cells(img)                                               # Draw cells on image
            cv2.imshow(image.get_name(), cv2.resize(img, (0, 0),
                                                    fx=0.25, fy=0.25))          # Resize and show image
            cv2.waitKey(0)                                                      # Wait for keypress
            cv2.destroyAllWindows()                                             # Destroy window
            for i, cell in enumerate(image.get_cells()):                        # For every cell
                img = self._crop_to_cell(image.get_image(), cell,
                                         dx=image_width, dy=image_height)       # Crop to cell centre and a given size
                cv2.imshow("Cell", img)                                         # Show cell to user
                cv2.waitKey(50)                                                 # Give the image time to load
                print("Classify the cell:")
                for j, label in enumerate(labels):                              # For each option
                    print("\t" + str(j + 1) + ". " + label[0] + " - " + label)  # Print option
                selection = input("")                                           # Get user input
                for label in labels:
                    if selection == label[0] or selection == str(labels.index(label) + 1):
                        image_name = destination + "/" + label + "/" \
                                     + image.get_name() + "_" + str(i) + "." + image.get_type()
                        print("Saving to " + image_name + ".")
                        cv2.imwrite(image_name, img)                            # Save image
                if selection == len(labels) + 1 or selection == "quit":
                    break
            cv2.destroyAllWindows()                                             # Destroy window

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
    def _make_cell_dirs(destination, labels):
        dirs = destination.split("/")
        path = ""
        for directory in dirs:
            if not os.path.isdir(path + directory):
                os.mkdir(path + directory)
            path += directory + "/"
        for directory in labels:
            if not os.path.isdir(path + directory):
                os.mkdir(path + directory)

    @staticmethod
    def _crop_to_cell(image, cell, dx=0, dy=0):
        if not dx:
            dx = cell.get_radius() * 2
        if not dy:
            dy = cell.get_radius() * 2
        height, width, _ = image.shape                                          # Get height and width of original image
        x1 = int(max(cell.get_position()[0] - dx / 2, 0))                       # Calculate bounding box coordinates
        y1 = int(max(cell.get_position()[1] - dy / 2, 0))
        x2 = int(min(cell.get_position()[0] + dx / 2, width))
        y2 = int(min(cell.get_position()[1] + dy / 2, height))
        return image[y1:y2, x1:x2, :]                                           # Return a cropped image

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
        if not isinstance(directories, list):
            directories = [directories]
        for directory in directories:                                           # For each directory
            if isinstance(directory, str):
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

    @staticmethod
    def _get_yaml_dict(path):
        yaml_dict = {}
        if os.path.isfile(path):
            with open(path) as file:
                yaml_dict = yaml.load(file)
        return yaml_dict
