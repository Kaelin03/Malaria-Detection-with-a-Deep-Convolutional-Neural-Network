#!/usr/bin/env python3

import os
import cv2
import yaml

from Image import Image
from CellDetector import CellDetector


class ManualClassifier(object):

    def __init__(self, yaml_path):
        """
        Initialises all Class attributes and updates them with values given in the config file
        :param yaml_path: path to the configuration file
        """
        self._yaml_path = "../config/" + yaml_path
        self._labels = []
        self._image_shape = None
        self._train_images = []
        self._test_images = []
        self._train_destination = None
        self._test_destination = None
        self._cell_detector = CellDetector(yaml_path)
        self.update_config()

    def update_config(self):
        """
        Updates all attributes as per the given configuration file
        :return: None
        """
        try:
            config = self._get_yaml_dict(self._yaml_path)["manual_classifier"]  # Get configurations from the yaml path
            self._image_shape = (config["image_width"], config["image_height"]) # Get the desired cell image size
            self._labels = config["labels"]                                     # Get the possible labels
            self._train_images = self._get_images(config["train"]["files"])     # Get the train images
            self._test_images = self._get_images(config["test"]["files"])       # Get the test images
            self._train_destination = "../" + config["train"]["destination"]    # Destination will start one level above
            self._test_destination = "../" + config["test"]["destination"]      # Destination will start one level above
        except KeyError:
            print("Warning: Some entries were missing from the manual_classifier configurations.")

    def run(self):
        """
        Calls methods to present the user with images of cells for manual classification
        :return: None
        """
        self.update_config()
        if self._train_images:
            self._detect_and_show(self._train_images, self._train_destination)
        if self._test_images:
            self._detect_and_show(self._test_images, self._test_destination)

    def _detect_and_show(self, images, destination):
        """
        :param images: a list of Image objects
        :param destination: directory to save images in
        :return:
        """
        for image in images:
            cells = self._cell_detector.run(image)
            image.add_cells(cells)
            img = image.get_image()
            image.draw_cells(img)
            img = cv2.resize(img, (0, 0), fx=0.25, fy=0.25)                         # Resize to 1/4 scale
            cv2.imshow(image.get_name(), img)                                       # Display image
            cv2.waitKey(0)                                                          # Show image and let it load
            cv2.destroyWindow(image.get_name())                                     # Destroy image window
            for i, cell in enumerate(image.get_cells()):
                img = cell.get_image(image.get_image())                             # Get the image of the cell
                cv2.imshow("Cell", img)                                             # Show the cell
                cv2.waitKey(50)                                                     # Give it time to load
                selection = self._get_classification(self._labels)                  # Get user classification
                self._save_cell(img, selection, destination, image, i)              # Save cell
                if selection == "q" or selection == str(len(self._labels) + 1):     # If user selects quit
                    break
            cv2.destroyWindow("Cell")                                               # Destroy cell window

    def _save_cell(self, img, selection, destination, image, nb):
        """
        Saves a given image to a given destination as a given name for the user's classification
        :param selection: user's classification of image
        :param destination: destination directory for image
        :param image: the image
        :param image_name: name for the image
        :return: None
        """
        saved = False
        image_name = image.get_name() + "_" + str(nb) + "." + image.get_type()      # Make image name
        for label in self._labels:
            cond_1 = selection == label[0]                                          # Selection matches first letter
            cond_2 = selection == str(self._labels.index(label) + 1)                # Selection matches label index
            if cond_1 or cond_2:
                image_dir = destination + "/" + label + "/" + image.get_name()      # Directory to save image in
                self._save_image(img, image_name, image_dir)                        # Save image
                print("Saving to " + image_dir + "/" + image_name)                  # Feedback
                saved = True                                                        # Flag that image has been saved
        if not saved and selection != "q" and selection != str(len(self._labels) + 1):
            image_dir = destination + "/unused/" + image.get_name()                 # Destination is unused folder
            self._save_image(img, image_name, image_dir)                            # Save image
            print("Saving to " + image_dir + "/" + image_name)                      # Feedback

    def _get_images(self, files):
        """
        Returns a list of image objects after removing any non jpg/png files
        :param: list of file paths
        :return: list of Image objects
        """
        images = []
        if files is not None:
            files = ["../" + file for file in files]                                # File paths start one level above
            files = self._check_images(files)                                       # Remove non image files
            images = [Image(file) for file in files]                                # Create image objects
        return images
    
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
            else:
                print("Warning: " + path + " not found.")
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
    def _get_classification(labels):
        """
        Displays the possible labels
        :param labels: possible labels
        :return: users selection
        """
        print("Classify the cell:")
        for j, label in enumerate(labels):                                          # For each label
            print("\t" + str(j + 1) + ". " + label[0] + " - " + label)              # Print options
        print("\t" + str(len(labels) + 1) + ". q - quit")                           # Show option to quit
        return input("")

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
