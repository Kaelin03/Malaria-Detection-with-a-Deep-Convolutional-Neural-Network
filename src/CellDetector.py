#!/usr/bin/env python3

import cv2
import math

from Cell import Cell


class CellDetector(object):

    def __init__(self, config):
        self._kernel_size = tuple([config["kernel_size"]] * 2)
        self._area_threshold = config["area_threshold"]
        self._min_rad = int((config["radius"] - config["tolerance"]) * config["pixels_per_micrometer"])
        self._max_rad = int((config["radius"] + config["tolerance"]) * config["pixels_per_micrometer"])
        if config["minimum_distance"]:
            self._min_dist = int(config["minimum_distance"])
        else:
            self._min_dist = self._min_rad

    def run(self, image):
        cells = []
        print("Detecting cells in " + image.get_name() + "...")
        img = image.get_image()                                                 # Get the image
        img = self._pre_process(img)                                            # Pre-process for Hough Circles
        circles = self._hough_circles(img)                                      # Detect circles
        for circle in circles:
            coverage = self._get_coverage(img, circle)                          # Check coverage of each circle
            if coverage > self._area_threshold:                                 # If coverage is above the threshold
                cells.append(Cell(circle[0:2], circle[2]))                      # Append a new cell object
        print(str(len(cells)) + " cells found.")
        return cells

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
                                   minDist=self._min_dist,
                                   param1=7,
                                   param2=6,
                                   minRadius=self._min_rad,
                                   maxRadius=self._max_rad)[0]
        circles = circles.tolist()                                                  # Convert np array to list
        return circles                                                              # Return list of circles

    @staticmethod
    def _get_coverage(img, circle):
        # Given an image and a circle
        # Returns the percentage of pixels in the circle that are black
        height, width = img.shape                                                   # Dimensions of image
        x = int(circle[0])                                                          # X-coordinate of circe centre
        y = int(circle[1])                                                          # Y-coordinate of circle centre
        rad = int(circle[2])                                                        # Radius of circle
        black = 0                                                                   # Total number of black pixels
        non_black = 0                                                               # Total number of non-black pixels
        for i in range(x - rad, x + rad + 1):                                       # For each pixel in the x-direction
            for j in range(y - rad, y + rad + 1):                                   # For each pixel in the y-direction
                in_image = 0 <= j < height and 0 <= i < width                       # The point is in the image
                in_circle = pythagoras((i - x), (j - y)) < rad                      # The point is in the circle
                if in_image and in_circle:                                          # If in the image and in the circle
                    if img[j][i] == 0:                                              # If point is black
                        black += 1                                                  # Add to total black pixels
                    else:                                                           # If point is not black
                        non_black += 1                                              # Add one to total non_black pixels
        percentage = black / float(black + non_black) * 100                         # Percentage of black pixels
        return percentage


def pythagoras(a, b):
    # Given two numbers, returns pythagoras
    c = math.sqrt(a ** 2 + b ** 2)
    return c


