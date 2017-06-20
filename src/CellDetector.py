#!/usr/bin/env python3

import os
import cv2
import yaml

# from inspect import currentframe, getframeinfo
# frameinfo = getframeinfo(currentframe())
# print frameinfo.filename, frameinfo.lineno


class CellDetector(object):

    def __init__(self, config_name=""):
        self._rbc = []                                  # List of detected red blood cells
        self._wbc = []                                  # List of detected white blood cells
        self._kernel_size = None                        # Tuple of odd integers for Gaussian blur kernel
        self._min_rbc_rad = None                        # Minimum expected radius of red blood cells
        self._max_rbc_rad = None                        # Maximum expected radius of red blood cells
        self._min_rbc_dist = None                       # Minimum accepted distance between red blood cell centres
        self._configured = [False] * 3                  # Flag to notify if the detector is configured or not

    def configure(self, config_path):
        # Given a yaml of parameters
        # Reads the config yaml and updates the parameters accordingly
        config_path = "config/" + config_path                               # Config file will be in config directory
        if os.path.isfile(config_path):                                     # Check config file exists
            with open(config_path) as file:                                 # Open file
                config_dict = yaml.load(file)                               # Load data into a dict
            try:
                self.set_kernel_size(config_dict["kernel_size"])            # Set kernel size
            except KeyError:
                print("Warning: kernel_size not found in " + config_path + ".")
            try:
                self.set_rbc_rad(config_dict["rbc_rad"][0], config_dict["rbc_tol"][0])      # Set rbc radii
            except KeyError:
                print("Warning: rbc_rad and/or rbc_tol not found in " + config_path + ".")
            try:
                self.set_rbc_min_dist(config_dict["rbc_min_dist"][0])                       # Set rbc_min_dist
            except KeyError:
                print("Warning: rbc_min_dist not found in " + config_path + ".")
            if all(self._configured):
                print("Successfully configured Cell Detector.")
        else:
            print("Warning: " + config_path + " is not a file.")
            print("Warning: Cell Detector not configured.")

    def manually_classify_circles(self, image_path):
        # Given a path to an image
        # Displays each circle detected in sequence for manual classification
        if all(self._configured):
            image = cv2.imread(image_path)
            print("Performing circle detection on " + image_path + ".")
            circles = self._hough_circles(image)
            for circle in circles:
                cv2.circle(image, tuple(map(int, circle[0:2])), int(circle[2]), (0, 255, 0), 2)
            cv2.imshow("circles", image)
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        else:
            print("Warning: cannot manually classify circles, Cell Detector not configured.")

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

    def set_rbc_rad(self, radius, tol):
        # Given a radius and a tolerance
        # Sets the max and min rbc radii
        try:
            self._max_rbc_rad = int((radius + tol / 2.0) * 10.0)
            self._min_rbc_rad = int((radius - tol / 2.0) * 10.0)
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
