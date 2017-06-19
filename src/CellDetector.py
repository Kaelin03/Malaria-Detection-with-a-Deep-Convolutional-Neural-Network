#!/usr/bin/env python3

import os
import cv2
import yaml

# from inspect import currentframe, getframeinfo
# frameinfo = getframeinfo(currentframe())
# print frameinfo.filename, frameinfo.lineno


class CellDetector(object):

    def __init__(self, config_name=""):
        self._rbc = None
        self._wbc = None
        self._kernel_size = None
        self._min_rbc_rad = None
        self._max_rbc_rad = None
        self._min_rbc_dist = None
        self._rbc_tolerance = None
        self._config_path = "config/" + config_name
        self._configured = False
        self.configure()

    def configure(self):
        if self._config_path:
            if os.path.isfile(self._config_path):
                with open("config/cell_detector.yaml") as file:
                    config_dict = yaml.load(file)
                try:
                    self._set_parameters(config_dict)
                    print("Cell Detector configured.")
                except TypeError:
                    print("Warning: Cell Detector not configured.")
            else:
                print("Warning: " + self._config_path + " is not a file")

    def manually_classify_circles(self, image_path):
        # Given a path to an image
        # Displays each circle detected in sequence for manual classification
        image = cv2.imread(image_path)
        circles = self._hough_transform(image)

    def _hough_transform(self, image):
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
        return circles

    def find_wbc(self, image_path):
        pass

    def find_rbc(self, image_path):
        pass

    def _set_parameters(self, config_dict):
        self._kernel_size = tuple(config_dict["kernel_size"])
        rbc_tol = config_dict["rbc_tolerance"][0]
        rbc_rad = config_dict["rbc_radius"][0]
        self._min_rbc_rad = int((rbc_rad - rbc_tol / 2.0) * 10.0)
        self._max_rbc_rad = int((rbc_rad + rbc_tol / 2.0) * 10.0)
        self._min_rbc_dist = float(config_dict["rbc_radius"][0])
        if self._min_rbc_dist < 0:
            self._min_rbc_dist = self._min_rbc_rad
        self._configured = True