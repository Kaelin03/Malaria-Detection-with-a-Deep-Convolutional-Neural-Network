#!/usr/bin/env python3

import os
import cv2
import yaml
import copy
import numpy as np

import src.helpers as helpers
from sklearn import svm

# from inspect import currentframe, getframeinfo
# frameinfo = getframeinfo(currentframe())
# print frameinfo.filename, frameinfo.lineno


class CellDetector(object):

    def __init__(self):
        self._rbc = []                                  # List of detected red blood cells
        self._wbc = []                                  # List of detected white blood cells
        self._kernel_size = None                        # Tuple of odd integers for Gaussian blur kernel
        self._min_rbc_rad = None                        # Minimum expected radius of red blood cells
        self._max_rbc_rad = None                        # Maximum expected radius of red blood cells
        self._min_rbc_dist = None                       # Minimum accepted distance between red blood cell centres
        self._rbc_positives = None                      # Path to directory of rbc images
        self._rbc_negatives = None                      # Path to directory of non-rbc images
        self._features = 50                             # Number of bins in each colour histogram
        self._configured = [False] * 4                  # Flag to notify if the detector is configured or not
        self._clf = svm.SVC(gamma=0.001, C=100)         # Initialise SVM

    def configure(self, config_path):
        # Given a yaml of parameters
        # Reads the config yaml and updates the parameters accordingly
        print("Configuring Cell Detector...")
        config_path = "config/" + config_path                               # Config file will be in config directory
        if os.path.isfile(config_path):                                     # Check config file exists
            with open(config_path) as file:                                 # Open file
                config_dict = yaml.load(file)                               # Load data into a dict
            try:
                self.set_kernel_size(config_dict["kernel_size"])            # Set kernel size
            except KeyError:
                print("Warning: kernel_size not found in " + config_path + ".")
            try:
                self.set_rbc_rad(config_dict["rbc_rad"][0],
                                 config_dict["rbc_tol"][0],
                                 config_dict["pixels_per_micrometer"][0])   # Set rbc radii
            except KeyError:
                print("Warning: rbc_rad/ rbc_tol/ pixels_per_micrometer not found in " + config_path + ".")
            try:
                self.set_rbc_min_dist(config_dict["rbc_min_dist"][0])       # Set rbc_min_dist
            except KeyError:
                print("Warning: rbc_min_dist not found in " + config_path + ".")
            try:
                self.set_rbc_paths(config_dict["rbc_positives"][0],
                                   config_dict["rbc_negatives"][0])
            except KeyError:
                print("Warning: rbc_positives/ rbc_negatives not found in " + config_path + ".")
            if all(self._configured):
                print("Successfully configured Cell Detector.")
        else:
            print("Warning: " + config_path + " is not a file.")
            print("Warning: Cell Detector not configured.")

    def manually_classify_circles(self, samples):
        # Given a list of samples
        # Displays each circle detected in sequence for manual classification
        if all(self._configured[0:3]):                                              # If adequately configured
            if samples:                                                             # If some samples exist
                quit_flag = False
                self._create_rbc_training_directories()
                for sample in samples:                                              # For each sample
                    if quit_flag:                                                   # If the user wants to quit
                        break
                    for image in sample.get_images():                               # For each image in each sample
                        if quit_flag:                                               # If the user wants to quit
                            break
                        img = cv2.imread(image.get_path())                          # Get the image from the path
                        print("Performing circle detection on " + image.get_path() + "...")
                        circles = self._hough_circles(img)                          # Hough circle transform
                        self._show_numbered_circles(copy.deepcopy(img), circles)
                        option = 0                                                  # Reset option to zero200
                        for n, circle in enumerate(circles):                        # For each circle detected
                            try:
                                option = int(option)
                            except ValueError:
                                option = 0
                            if n >= option:
                                cropped_img = helpers.crop_to_circle(img, circle)  # Crop the image to the circle
                                cv2.imshow("circles", cropped_img)  # Show the cropped image
                                cv2.waitKey(10)  # Give the image time to load
                                print("Showing cell " + str(n) + ".")
                                option = input("""Classify the cell: 
        - Enter <p> if image is a cell.
        - Enter <n> if the image is not a cell.
        - Enter any other key to ignore this image.
        - Enter <next> to stop classifying this image.
        - Enter <quit> to quit classifying.
        - Enter a number to skip forward to a cell.\n""")
                                cropped_img_name = image.get_name() + "_" + str(n) + "." + image.get_type()
                                if option == 'p':
                                    cv2.imwrite(self._rbc_positives + "/" + cropped_img_name, cropped_img)
                                elif option == 'n':
                                    cv2.imwrite(self._rbc_negatives + "/" + cropped_img_name, cropped_img)
                                elif option == "next":
                                    break
                                elif option == "quit":
                                    quit_flag = True
                                    break
            else:
                print("Warning: no training samples given.")
        else:
            print("Warning: cannot manually classify circles, Cell Detector not configured.")

    def train(self):
        if self._configured[3]:                         # If adequately configured
            print("Reading training images...")
            if len(os.listdir(self._rbc_positives)) == 0:
                print("Warning: no training samples found in " + self._rbc_positives)
            elif len(os.listdir(self._rbc_negatives)) == 0:
                print("Warning: no training samples found in " + self._rbc_positives)
            else:
                images = []
                labels = []
                for file in os.listdir(self._rbc_positives):
                    if os.path.isfile(self._rbc_positives + "/" + file):
                        images.append(cv2.imread(self._rbc_positives + "/" + file))     # Append train image to images
                        labels.append("positive")                                       # Append label to labels
                for file in os.listdir(self._rbc_negatives):
                    if os.path.isfile(self._rbc_negatives + "/" + file):
                        images.append(cv2.imread(self._rbc_negatives + "/" + file))
                        labels.append("negative")
                print("Read " + str(len(images)) + " training images.")
                x, y = self._assemble_data(images, labels)
                print("Training support vector machine...")
                self._clf.fit(x, y)
                print("Successfully trained support vector machine.")
        else:
            print("Warning: cannot train for RBCs, Cell Detector not configured.")

    def detect_rbc(self, samples):
        for sample in samples:
            for image in sample.get_images():
                img = cv2.imread(image.get_path())                              # Get the image from the path
                print("Performing circle detection on " + image.get_path() + "...")
                circles = self._hough_circles(img)                              # Hough circle transform
                for circle in circles:
                    cropped_img = helpers.crop_to_circle(img, circle)
                    hist = np.concatenate((self._get_histogram(cropped_img, 0),
                                           self._get_histogram(cropped_img, 1),
                                           self._get_histogram(cropped_img, 2)), axis=1)
                    prediction = self._clf.predict(hist)
                    if prediction == "negative":
                        cv2.circle(img, tuple(map(int, circle[0:2])), int(circle[2]), (0, 0, 255), 2)
                    elif prediction == "positive":
                        cv2.circle(img, tuple(map(int, circle[0:2])), int(circle[2]), (0, 255, 0), 2)
                    else:
                        print("Warning: category " + prediction + " unknown.")
                cv2.imshow("image", img)
                cv2.waitKey(0)
                cv2.destroyAllWindows()

    def _hough_circles(self, image):
        # Given an image
        # Performs Hough circle transform and returns a list of positions and radii
        circles = []
        if self._configured:
            image_gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)                        # Convert to grayscale
            image_gray = cv2.GaussianBlur(image_gray, self._kernel_size, 0)             # Gaussian blur
            _, image_threshold = cv2.threshold(image_gray, 0, 255,
                                               cv2.THRESH_BINARY + cv2.THRESH_OTSU)     # Binary threshold
            circles = cv2.HoughCircles(image_threshold, cv2.HOUGH_GRADIENT, 1,          # Hough circle transform
                                       minDist=self._min_rbc_dist,
                                       param1=7,
                                       param2=6,
                                       minRadius=self._min_rbc_rad,
                                       maxRadius=self._max_rbc_rad)[0]
            circles = circles.tolist()                                                  # Convert np array to list
        else:
            print("Warning: CellDetector is not configured")
        print(str(len(circles)) + " circles detected.")
        return circles

    def set_kernel_size(self, kernel_size):
        # Given a list to describe the kernel size
        # Sets the kernel size
        try:
            self._kernel_size = [int(value) for value in kernel_size]       # Convert each item to int
            if len(self._kernel_size) == 1:                                 # If only one value is given
                self._kernel_size *= 2                                      # Double the list
            self._kernel_size = tuple(self._kernel_size)                    # Make tuple
            self._configured[0] = True                                      # Signify that the kernel has been set
        except ValueError:
            print("Warning: invalid kernel size given.")
            self._configured[1] = False                                     # Signify that the kernel has not been set

    def set_rbc_rad(self, radius, tol, ppm):
        # Given a radius and a tolerance
        # Sets the max and min rbc radii
        try:
            self._max_rbc_rad = int((radius + tol) * float(ppm))
            self._min_rbc_rad = int((radius - tol) * float(ppm))
            self._configured[1] = True
        except ValueError:
            print("Warning: invalid rbc values given.")
            self._configured[1] = False

    def set_rbc_min_dist(self, dist):
        try:
            self._min_rbc_dist = int(dist)                              # Convert to int
            self._configured[2] = True                                  # Signify that the value has been set
            if self._min_rbc_dist <= 0:
                if self._configured[1]:
                    self._min_rbc_dist = self._min_rbc_rad              # Set min dist = min rad
                    self._configured[2] = True                          # Signify that the value has been set
                else:
                    print("Warning: min rbc distance could not be automatically set.")
                    print("Ensure that rbc radii are properly configured.")
                    self._configured[2] = False                         # Signify that the value has not been set
        except ValueError:
            print("Warning: min rbc dist not set.")
            self._configured[2] = False                                 # Signify that the value has not been set

    def set_rbc_paths(self, positives, negatives):
        self._rbc_positives = str(positives)
        self._rbc_negatives = str(negatives)
        self._configured[3] = True

    def _create_rbc_training_directories(self):
        positive_path = self._rbc_positives.split("/")
        negative_path = self._rbc_negatives.split("/")
        self._make_directories(positive_path)
        self._make_directories(negative_path)

    def _assemble_data(self, images, labels):
        print("Assembling training data...")
        x = np.empty((0, self._features * 3))
        y = np.empty(0, dtype=str)
        for image, label in zip(*(images, labels)):
            hist = np.concatenate((self._get_histogram(image, 0),
                                   self._get_histogram(image, 1),
                                   self._get_histogram(image, 2)), axis=1)
            x = np.append(x, hist, axis=0)              # Append the image features (histogram) to the x array
            y = np.append(y, label)                     # Append the image label to the y array
        print("Successfully assembled training data...")
        return x, y

    def _get_histogram(self, image, channel):
        hist = cv2.calcHist([image], [channel], None, [self._features], [0, 255])
        cv2.normalize(hist, hist, 0, 255, cv2.NORM_MINMAX)                      # Normalise histogram between 0 and 255
        hist = np.reshape(hist, (1, self._features))                            # Reshape histogram
        return hist

    @staticmethod
    def _show_numbered_circles(img, circles):
        for n, circle in enumerate(circles):
            cv2.circle(img, tuple(map(int, circle[0:2])), int(circle[2]), (0, 0, 0), 2)
            cv2.putText(img, str(n), tuple(map(int, circle[0:2])), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 0))
        cv2.imshow("image", img)
        cv2.waitKey(0)
        cv2.destroyAllWindows()


    @staticmethod
    def _make_directories(directories):
        path = ""
        for directory in directories:
            if not os.path.isdir(path + directory):
                os.mkdir(path + directory)
            path += directory + "/"
