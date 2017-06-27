#!/usr/bin/env python3

import os
import cv2
import yaml
import math
import numpy as np

from Cell import Cell


class CellDetector(object):

    def __init__(self, config):
        self._rbc = []                                  # List of detected red blood cells
        self._wbc = []                                  # List of detected white blood cells
        self._kernel_size = None                        # Tuple of odd integers for Gaussian blur kernel
        self._min_rbc_rad = None                        # Minimum expected radius of red blood cells
        self._max_rbc_rad = None                        # Maximum expected radius of red blood cells
        self._min_rbc_dist = None                       # Minimum accepted distance between red blood cell centres
        self._area_threshold = None                     # Percentage of the circle that must be filled
        self._configure(config)                         # Set values as per the given config path

    def run(self, image):
        cells = []
        if all([self._kernel_size,
                self._min_rbc_rad,
                self._max_rbc_rad,
                self._min_rbc_dist,
                self._area_threshold]):                                             # Check all parameters are set
            print("Detecting cells in " + image.get_name() + "...")
            img = image.get_image()                                                 # Get the image
            img = self._pre_process(img)                                            # Pre-process for Hough Circles
            circles = self._hough_circles(img)                                      # Detect circles
            for circle in circles:
                coverage = get_coverage(img, circle)                                # Check coverage of each circle
                if coverage > self._area_threshold:                                 # If coverage is above the threshold
                    cells.append(Cell(circle[0:2], circle[2]))                      # Append a new cell object
            print(str(len(cells)) + " cells found.")
        else:
            print("Warning: cell detector not properly configured.")
        return cells

    def _configure(self, config):
        # Given the name of the cell detector configuration file
        # Sets each parameter of the cell detector
        print("Configuring cell detector...")
        config_path = "../config/" + config
        if os.path.isfile(config_path):
            with open(config_path) as file:
                config_dict = yaml.load(file)
            try:
                self._set_kernel_size(config_dict["kernel_size"])                   # Set kernel size
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: kernel_size not found.")
            try:
                self._set_rbc_rad(config_dict["rbc_radius"][0],
                                  config_dict["rbc_tolerance"][0],
                                  config_dict["pixels_per_micrometer"][0])          # Set rbc_rad
            except KeyError:                                                        # Key may not be in config_dict
                print("""Warning: either:
        - rbc_radius,
        - rbc_tolerance or,
        - pixels_per_micrometer not found.""")
            try:
                self._set_rbc_min_dist(config_dict["rbc_min_distance"][0])          # Set rbc_min_dist
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: rbc_min_distance not found.")
            try:
                self._set_area_threshold(config_dict["area_threshold"][0])          # Set area_threshold
            except KeyError:                                                        # Key may not be in config_dict
                print("Warning: area_threshold not found.")
            print("Done.")
        else:
            print("Warning: " + config + " not found.")

    def _set_kernel_size(self, kernel_size):
        # Given a list to describe the kernel size
        # Sets the kernel size
        try:
            self._kernel_size = [int(value) for value in kernel_size]               # Convert each item to int
            if len(self._kernel_size) == 1:                                         # If only one value is given
                self._kernel_size *= 2                                              # Double the list
            self._kernel_size = tuple(self._kernel_size)                            # Make tuple
        except ValueError:
            self._kernel_size = None                                                # Signify kernel_size not set
            print("Warning: invalid kernel size given.")

    def _set_rbc_rad(self, radius, tol, ppm):
        # Given a radius and a tolerance
        # Sets the max and min rbc radii
        try:
            self._max_rbc_rad = int((radius + tol) * float(ppm))                    # Given radius plus tolerance
            self._min_rbc_rad = int((radius - tol) * float(ppm))                    # Given radius minus tolerance
        except ValueError:
            self._max_rbc_rad = None                                                # Signify max_rbc_rad is not set
            self._min_rbc_rad = None                                                # Signify min_rbc_rad is not set
            print("Warning: invalid rbc values given.")

    def _set_rbc_min_dist(self, dist):
        try:
            self._min_rbc_dist = int(dist)                                          # Convert to int
            if self._min_rbc_dist <= 0:                                             # If given value is <= 0
                if self._min_rbc_rad is not None:                                   # If min_rbc_rad has been set
                    self._min_rbc_dist = self._min_rbc_rad                          # Set min dist = min rad
                else:
                    self._min_rbc_dist = None                                       # Signify min_rbc_dist is not set
                    print("Warning: min_rbc_distance could not be set.")
                    print("Ensure that rbc radius is set.")
        except ValueError:
            self._min_rbc_dist = None                                               # Signify min_rbc_dist is not set
            print("Warning: min_rbc_distance not set.")

    def _set_area_threshold(self, area):
        try:
            self._area_threshold = float(area)                                      # Ensure the threshold is a float
        except ValueError:
            self._area_threshold = None                                             # Signify area_threshold is not set
            print("Warning: area_threshold not set.")

    def _pre_process(self, img):
        # Given a BGR numpy array
        # Returns a binary image for Hough Circle Transform
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)                            # Convert to gray scale
        img_gray = cv2.GaussianBlur(img_gray, self._kernel_size, 0)                 # Gaussian blur
        _, img_threshold = cv2.threshold(img_gray, 0, 255,
                                         cv2.THRESH_BINARY + cv2.THRESH_OTSU)       # Binary threshold
        return img_threshold                                                        # Return the binary image

    def _hough_circles(self, img):
        # Given a binary image
        # Performs Hough circle transform and returns a list of positions and radii
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,                      # Hough circle transform
                                   minDist=self._min_rbc_dist,
                                   param1=7,
                                   param2=6,
                                   minRadius=self._min_rbc_rad,
                                   maxRadius=self._max_rbc_rad)[0]
        circles = circles.tolist()                                                  # Convert np array to list
        return circles                                                              # Return list of circles


def pythagoras(a, b):
    # Given two numbers, returns pythagoras
    c = math.sqrt(a ** 2 + b ** 2)
    return c


def get_coverage(img, circle):
    # Given an image and a circle
    # Returns the percentage of pixels in the circle that are black
    height, width = img.shape                                                       # Dimensions of image
    x = int(circle[0])                                                              # X-coordinate of circe centre
    y = int(circle[1])                                                              # Y-coordinate of circle centre
    rad = int(circle[2])                                                            # Radius of circle
    black = 0                                                                       # Total number of black pixels
    non_black = 0                                                                   # Total number of non-black pixels
    for i in range(x - rad, x + rad + 1):                                           # For each pixel in the x-direction
        for j in range(y - rad, y + rad + 1):                                       # For each pixel in the y-direction
            in_image = 0 <= j < height and 0 <= i < width                           # The point is in the image
            in_circle = pythagoras((i - x), (j - y)) < rad                          # The point is in the circle
            if in_image and in_circle:                                              # If in the image and in the circle
                if img[j][i] == 0:                                                  # If point is black
                    black += 1                                                      # Add to total black pixels
                else:                                                               # If point is not black
                    non_black += 1                                                  # Add one to total non_black pixels
    percentage = black / float(black + non_black) * 100                             # Percentage of black pixels
    return percentage
