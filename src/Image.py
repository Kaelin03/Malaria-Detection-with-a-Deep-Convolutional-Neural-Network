#!/usr/bin/env python3

import cv2


class Image(object):

    def __init__(self, path):
        self._path = path
        self._name = path.split("/")[-1].split(".")[-2]     # Extract the name of the image from that path
        self._type = path.split("/")[-1].split(".")[-1]     # Extract the image type from the path
        self._id = self._name.split("_")[-1]                # Extract the image id number from the name
        self._cells = []

    def add_cell(self, cell):
        # Add a new cell object to cells
        self._cells.append(cell)

    def get_cells(self):
        # Returns list of cell objects associated with the image
        return self._cells

    def get_id(self):
        # Returns the name of the sample
        return self._id

    def get_path(self):
        # Returns the path to the image
        return self._path

    def get_name(self):
        # Returns the name of the image
        return self._name

    def get_type(self):
        # Returns the image type
        return self._type

    def get_image(self):
        # Returns the image as a BGR numpy array
        return cv2.imread(self._path)
