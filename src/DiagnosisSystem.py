#!/usr/bin/env python3

import os
import cv2
import yaml
import random
import numpy as np

from Image import Image
from CellDetector import CellDetector
from Classifier import Classifier


class DiagnosisSystem(object):

    def __init__(self, yaml_path):
        self._yaml_path = "../config/" + yaml_path
        yaml_dict = self._get_yaml_dict(self._yaml_path)                            # Get dict from given yaml file name
        self._manual_config = yaml_dict["manual_classifier"]                        # Config for manual classifier
        self._train_config = yaml_dict["train"]                                     # Config for training
        self._diagnose_config = yaml_dict["diagnose"]                               # Config for diagnosing
        self._cell_detector = CellDetector(yaml_dict["cell_detector"])              # Init cell detector
        self._classifier = Classifier(yaml_dict["classifier"])                      # Init classifier

    def manually_classify(self):
        train_paths = self._manual_config["train"]["files"]
        test_paths = self._manual_config["test"]["files"]
        if train_paths:                                                             # If train paths are given
            train_paths = ["../" + path for path in train_paths]                    # Start up one level
            train_paths = self._check_images(train_paths)                           # Check the paths are valid
            train_images = self._make_images(train_paths)                           # Convert paths into Images
            train_destination = self._manual_config["train"]["destination"]         # Destination of train cells
            self._detect_and_classify(train_images, train_destination)              # Detect and classify train cells
        if test_paths:                                                              # If test paths are given
            test_paths = ["../" + path for path in test_paths]                      # Start up one level
            test_paths = self._check_images(test_paths)                             # Check the paths are valid
            test_images = self._make_images(test_paths)                             # Convert paths into Images
            test_destination = self._manual_config["test"]["destination"]           # Destination of train cells
            self._detect_and_classify(test_images, test_destination)                # Detect and classify train cells

    def train(self):
        data = []
        shape = (self._train_config["image_width"],
                 self._train_config["image_height"],
                 self._train_config["image_depth"])                                 # Get image size
        for label in self._train_config["labels"]:                                  # For each image class
            print(label)
            directories = self._train_config["labels"][label]["directories"]
            if directories is not None:
                for directory in directories:                                       # For each directory given
                    data += self._get_images("../" + directory, label)              # Get images from directory
        if self._train_config["small_images"] == "ignore":
            data = self._ignore_small(data, shape)                                  # Remove small images
        elif self._train_config["small_images"] == "pad":
            data = self._pad_with(data, shape, 255)                                 # Pad small images with zeros
        else:
            print("No changes made to image shapes.")
        random.shuffle(data)                                                        # Shuffle the data
        x, y = self._to_arrays(data, shape)                                         # Put the data into numpy arrays
        x = self._resize(x, self._train_config["resize"])                           # Resize images
        x = self._normalise(x)
        cont = self._check_balance(y)                                               # Check that data is balanced
        if cont:
            self._classifier.train(x, y)                                            # Train the CNNs

    def load_model(self):
        self._classifier.load_model()

    def save_model(self):
        self._classifier.save_model()

    def plot_model(self):
        image_size = input("Enter the image dimensions:\n")
        image_size = tuple(map(int, image_size.split(",")))
        self._classifier.compile_model(image_size=image_size, num_classes=2)
        self._classifier.plot_model()

    def evaluate(self):
        evaluate_config = self._get_yaml_dict(self._yaml_path)["evaluate"]
        shape = (evaluate_config["image_width"],
                 evaluate_config["image_height"],
                 evaluate_config["image_depth"])                                    # Get image size
        num_classes = len(evaluate_config["labels"])
        for label in evaluate_config["labels"]:                                     # For each image class
            directories = evaluate_config["labels"][label]["directories"]
            if directories is not None:
                data = []
                for directory in directories:
                    data += self._get_images("../" + directory, label)
                if evaluate_config["small_images"] == "ignore":
                    data = self._ignore_small(data, shape)                              # Remove small images
                elif evaluate_config["small_images"] == "pad":
                    data = self._pad_with(data, shape, 255)                             # Pad small images with zeros
                else:
                    print("No changes made to image shapes.")
                x, y = self._to_arrays(data, shape)
                x = self._resize(x, self._train_config["resize"])                       # Resize images
                x = self._normalise(x)
                self._classifier.evaluate(x, y, num_classes=num_classes)

    def diagnose(self):
        # TODO get x from test samples
        # y = self._classifier.test(x)
        pass

    def _get_images(self, directory, label):
        data = []
        if os.path.isdir(directory):                                                # If the directory exists
            image_paths = os.listdir(directory)                                     # Get the file names in the dir
            image_paths = [directory + "/" + path for path in image_paths]          # Get the full file path
            image_paths = self._check_images(image_paths)                           # Check the files are images
            for path in image_paths:
                data.append([cv2.imread(path), label])                              # Append image and label to data
        else:
            print("Warning: " + directory + " not found.")
        return data

    def _detect_and_classify(self, images, destination):
        dx = self._manual_config["image_width"]                                     # Get height for cropping image
        dy = self._manual_config["image_height"]                                    # Get width for cropping image
        labels = self._manual_config["labels"]                                      # Labels of each class
        for image in images:                                                        # For each image
            cells = self._cell_detector.run(image)                                  # Detect cells
            image.add_cells(cells)                                                  # Attribute cells to the image
            img = image.get_image()                                                 # Get the cv2 image
            image.draw_cells(img)                                                   # Draw cells on the image
            cv2.imshow(image.get_name(), cv2.resize(img, (0, 0), fx=0.25, fy=0.25))
            cv2.waitKey(0)                                                          # Show image and let it load
            cv2.destroyAllWindows()                                                 # Destroy image window
            for i, cell in enumerate(image.get_cells()):                            # For each cell in the image
                img = self._crop_to_cell(image.get_image(), cell, dx=dx, dy=dy)     # Crop the image to the cell
                cv2.imshow("Cell", img)                                             # Show the cell
                cv2.waitKey(50)                                                     # Give it time to load
                selection = self._get_classification(self._manual_config["labels"]) # Get user classification
                saved = False
                image_name = image.get_name() + "_" + str(i) + "." + image.get_type()       # Make image name
                for label in labels:
                    if selection == label[0] or selection == str(labels.index(label) + 1):
                        image_dir = "../" + destination + "/" + label + "/" + image.get_name()      # Get directory
                        self._save_image(img, image_name, image_dir)                                # Save image
                        print("Saving to " + image_dir + "/" + image_name)
                        saved = True
                if selection == "q" or selection == str(len(labels) + 1):
                    break                                                           # Move on to next image
                elif not saved:                                                     # If not yet saved
                    image_dir = "../" + destination + "/unused/" + image.get_name()
                    self._save_image(img, image_name, image_dir)                    # Save to unused
                    print("Saving to " + image_dir + "/" + image_name)
            cv2.destroyAllWindows()                                                 # Destroy image window

    @staticmethod
    def _to_arrays(data, shape):
        y = np.empty((len(data), 1), dtype=np.uint8)  # Init labels array
        x = np.empty((len(data), shape[0], shape[1], shape[2]), dtype=np.uint8)  # Init data array
        for i, item in enumerate(data):                                             # Put data into numpy arrays
            x[i] = item[0]                                                          # x contains images
            y[i] = item[1]                                                          # y contains labels
        return x, y

    @staticmethod
    def _resize(x, size):
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
        classes = np.unique(y)                                                      # Find the number of classes
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
    def _pad_with(data, shape, value=0):
        padded_data = []
        for item in data:
            if item[0].shape != shape:
                x, y, z = item[0].shape
                dx = int((shape[0] - x) / 2)
                dy = int((shape[1] - y) / 2)
                tmp = np.zeros(shape)
                tmp.fill(value)
                tmp[dx:x+dx, dy:y+dy, 0:z] = item[0]
                item[0] = tmp
            padded_data.append(item)
        return padded_data

    @staticmethod
    def _ignore_small(data, shape):
        return [item for item in data if item[0].shape == shape]

    @staticmethod
    def _save_image(image, image_name, destination):
        # Given an image, the image name and a path
        # Checks the paths exists and creates it if not
        # Saves the image to the path with the given name
        directories = destination.split("/")                                        # Get list of directories
        path = ""                                                                   # Init path string
        for directory in directories:                                               # For each directory
            path += directory + "/"                                                 # Add it to the path
            if not os.path.isdir(path):                                             # If it does not yet exist
                os.mkdir(path)                                                      # Make it
        cv2.imwrite(path + "/" + image_name, image)                                 # Save image to path

    @staticmethod
    def _get_classification(labels):
        # Displays possible labels
        # Returns the user's selection
        print("Classify the cell:")
        for j, label in enumerate(labels):                                          # For each label
            print("\t" + str(j + 1) + ". " + label[0] + " - " + label)              # Print options
        print("\t" + str(len(labels) + 1) + ". q - quit")                           # Show option to quit
        selection = input("")
        return selection

    @staticmethod
    def _crop_to_cell(image, cell, dx=0, dy=0):
        if dx <= 0:                                                                 # If dx <= 0
            dx = cell.get_radius() * 2                                              # Crop to cell radius
        if dy <= 0:                                                                 # If dy <= 0
            dy = cell.get_radius() * 2                                              # Crop to cell radius
        height, width, _ = image.shape                                              # Get size of original image
        x1 = int(max(cell.get_position()[0] - dx / 2, 0))                           # Calculate bounding box values
        y1 = int(max(cell.get_position()[1] - dy / 2, 0))
        x2 = int(min(cell.get_position()[0] + dx / 2, width))
        y2 = int(min(cell.get_position()[1] + dy / 2, height))
        return image[y1:y2, x1:x2, :]                                               # Return a cropped image

    @staticmethod
    def _make_images(image_paths):
        images = [Image(path) for path in image_paths]                              # Create an Image obj for each path
        return images                                                               # Return a list of objects

    @staticmethod
    def _check_images(image_paths):
        # Given a list of file paths
        # Removes any non png/jpg files
        # Returns the list of file paths
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
            print("Waring: " + yaml_path + " not found.")
        return yaml_dict
