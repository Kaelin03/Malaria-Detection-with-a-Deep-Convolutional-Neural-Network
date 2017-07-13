#!/usr/bin/env python3

import os
import cv2
import math
import yaml

from Cell import Cell


class CellDetector(object):

    def __init__(self, yaml_path):
        """
        Initialises all Class attributes and updates them with values given in the config file
        :param yaml_path: path to the configuration file
        """
        self._yaml_path = "../config/" + yaml_path
        self._kernel = None
        self._area_threshold = None
        self._min_rad = None
        self._max_rad = None
        self._min_dist = None
        self.update_config()

    def update_config(self):
        """
        Updates all attributes as per the given configuration file
        :return: None
        """
        try:
            config = self._get_yaml_dict(self._yaml_path)["cell_detector"]      # Get configurations from the yaml path
            self._kernel = (config["kernel_size"], config["kernel_size"])       # Set the kernel size
            self._area_threshold = config["area_threshold"]                     # Set the area threshold
            self._min_rad = int((config["radius"] - config["tolerance"]) * config["pixels_per_micrometer"])
            self._max_rad = int((config["radius"] + config["tolerance"]) * config["pixels_per_micrometer"])
            if config["minimum_distance"]:                                      # If min dist is not zero
                self._min_dist = int(config["minimum_distance"])                # Set min dist
            else:
                self._min_dist = self._min_rad                                  # Set Min dis equal to min rad
        except KeyError:
            print("Warning: Some entries were missing from the cell_detector configurations.")

    def run(self, image):
        """
        :param: image: an image object
        :return: list of cell objects
        """
        self.update_config()
        cells = []                                                              # Initialise list of cell objects
        print("Detecting cells in " + image.get_name() + "...")                 # Feedback for user
        img = image.get_image()                                                 # Get the image
        img = self._pre_process(img)                                            # Pre-process for Hough Circles
        circles = self._hough_circles(img)                                      # Detect circles
        for circle in circles:                                                  # For every circle
            coverage = self._get_coverage(img, circle)                          # Check coverage of each circle
            if coverage > self._area_threshold:                                 # If coverage is above the threshold
                cells.append(Cell(circle[0:2], circle[2], image.get_path()))   # Append a new cell object
        print(str(len(cells)) + " cells found.")
        # except TypeError:
        #     print("Warning: cell detection failed.")
        #     print("Ensure all configuration values are correct.")
        return cells

    def _hough_circles(self, img):
        """
        Performs Hough Circle Transform on a binary image
        :param img: binary image
        :return: list of circle positions and radii
        """
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,                      # Hough circle transform
                                   minDist=self._min_dist,
                                   param1=7,
                                   param2=6,
                                   minRadius=self._min_rad,
                                   maxRadius=self._max_rad)[0]
        circles = circles.tolist()                                                  # Convert np array to list
        return circles

    def _pre_process(self, img):
        """
        :param img: BGR image
        :return: binary image
        """
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                        # Convert to gray scale
        img_gray = cv2.GaussianBlur(img_gray, self._kernel, 0)                  # Gaussian blur
        _, binary_img = cv2.threshold(img_gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)      # Binary threshold
        return binary_img

    def _get_coverage(self, img, circle):
        """
        Returns the percentage of an image in a given circle that is black
        :param img: binary image
        :param circle: list containing circle x, y and radius
        :return:
        """
        height, width = img.shape                                                   # Dimensions of image
        x = int(circle[0])                                                          # X-coordinate of circe centre
        y = int(circle[1])                                                          # Y-coordinate of circle centre
        rad = int(circle[2])                                                        # Radius of circle
        black = 0                                                                   # Total number of black pixels
        non_black = 0                                                               # Total number of non-black pixels
        for i in range(x - rad, x + rad + 1):                                       # For each pixel in the x-direction
            for j in range(y - rad, y + rad + 1):                                   # For each pixel in the y-direction
                in_image = 0 <= j < height and 0 <= i < width                       # The point is in the image
                in_circle = self._pythagoras((i - x), (j - y)) < rad                # The point is in the circle
                if in_image and in_circle:                                          # If in the image and in the circle
                    if img[j][i] == 0:                                              # If point is black
                        black += 1                                                  # Add to total black pixels
                    else:                                                           # If point is not black
                        non_black += 1                                              # Add one to total non_black pixels
        percentage = black / float(black + non_black) * 100                         # Percentage of black pixels
        return percentage

    @staticmethod
    def _pythagoras(a, b):
        """
        :param a: a length
        :param b: a length
        :return: pythagorean distance, c
        """
        c = math.sqrt(a ** 2 + b ** 2)
        return c

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
