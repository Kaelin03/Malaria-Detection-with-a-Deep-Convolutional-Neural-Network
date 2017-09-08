#!/usr/bin/env python3

import cv2
import numpy as np
from sklearn import svm
from sklearn.externals import joblib

from Cell import Cell
import helpers


class CellDetector(object):

    def __init__(self, yaml_path):
        self._yaml_path = "../config/" + yaml_path
        self._min_rad = None
        self._max_rad = None
        self._min_dist = None
        self._cell_shape = None
        self._clf = None
        self.update_config()

    def update_config(self):
        config = helpers.load_yaml(self._yaml_path)
        radius = float(config["cell_detector"]["radius"])
        tolerance = float(config["cell_detector"]["tolerance"])
        pixels_per_micrometer = float(config["cell_detector"]["pixels_per_micrometer"])
        image_height = config["images"]["height"]
        image_width = config["images"]["width"]
        image_depth = config["images"]["depth"]
        self._cell_shape = (image_height, image_width, image_depth)
        self._min_rad = int((radius - tolerance) * pixels_per_micrometer)
        self._max_rad = int((radius + tolerance) * pixels_per_micrometer)
        if config["cell_detector"]["minimum_distance"]:
            self._min_dist = int(config["minimum_distance"])
        else:
            self._min_dist = self._min_rad

    def run(self, image):
        self.update_config()
        if self._clf is None:
            print("Warning: SVM not compiled.")
        print("Detecting cells in " + image.get_name() + "...")
        img = image.get_image()
        circles = self._hough_circles(img)
        for circle in circles:
            if self._is_complete(img.shape, self._cell_shape, circle[0:2]):
                cell = Cell(circle[0:2], circle[2])
                if self._clf is not None:
                    cell_image = self._crop_to_cell(img, circle[0:2], 40)
                    x = self._get_pixels(cell_image, self._cell_shape)
                    p = self._clf.predict(x)[0]
                    if p == "1":
                        image.add_cell(cell)
                else:
                    image.add_cell(cell)

    def train(self, x, y):
        self.clear_model()
        self._clf.fit(x, y)

    def clear_model(self):
        self._clf = svm.SVC(gamma=0.001, C=100)

    def save_model(self, name):
        """
        :return:
        """
        joblib.dump(self._clf, name)

    def load_model(self):
        name = input("Enter the model name:\n")
        directory = "/".join(["../models", name])
        name = helpers.check_ext(name, "pkl")
        try:
            self._clf = joblib.load("/".join([directory, name]))
            print("Cell detector SVM successfully loaded.")
        except FileNotFoundError:
            print("Warning: " + "/".join([directory, name]) + " not found.")

    def _hough_circles(self, img):
        img = self._pre_process(img)
        circles = cv2.HoughCircles(img, cv2.HOUGH_GRADIENT, 1,
                                   minDist=self._min_dist,
                                   param1=7,
                                   param2=6,
                                   minRadius=self._min_rad,
                                   maxRadius=self._max_rad)[0]
        circles = circles.tolist()
        return circles

    @staticmethod
    def _get_pixels(img, shape):
        pixels = img.astype(np.float32)
        pixels /= 255
        pixels = pixels.reshape(1, np.prod(np.array(shape)))
        return pixels

    @staticmethod
    def _crop_to_cell(image, centre, radius):
        height, width, _ = image.shape
        x1 = int(max(centre[0] - radius, 0))
        y1 = int(max(centre[1] - radius, 0))
        x2 = int(min(centre[0] + radius, width))
        y2 = int(min(centre[1] + radius, height))
        return image[y1:y2, x1:x2, :]

    @staticmethod
    def _pre_process(img):
        img_gray = cv2.cvtColor(img, cv2.COLOR_RGB2GRAY)
        img_gray = cv2.GaussianBlur(img_gray, (5, 5), 0)
        _, binary_img = cv2.threshold(img_gray, 0, 255,
                                      cv2.THRESH_BINARY + cv2.THRESH_OTSU)
        return binary_img

    def _get_hist(self, img):
        hist = np.empty((1, 0))
        for ch in self._channels:
            h = cv2.calcHist([img], [ch], None, [self._bins], self._range)
            h.shape = (1, self._bins)
            hist = np.append(hist, h, axis=1)
        return hist

    @staticmethod
    def _is_complete(image_shape, cell_shape, cell_pos):
        if cell_pos[0] - cell_shape[1] / 2 < 0:
            return False
        elif cell_pos[1] - cell_shape[0] / 2 < 0:
            return False
        elif cell_pos[0] + cell_shape[1] / 2 > image_shape[1]:
            return False
        elif cell_pos[1] + cell_shape[0] / 2 > image_shape[0]:
            return False
        else:
            return True
