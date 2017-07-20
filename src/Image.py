#!/usr/bin/env python3

import cv2


class Image(object):

    def __init__(self, path):
        """
        :param path:
        """
        self._path = path
        self._name = path.split("/")[-1].split(".")[-2]     # Extract the name of the image from that path
        self._type = path.split("/")[-1].split(".")[-1]     # Extract the image type from the path
        self._id = self._name.split("_")[-1]                # Extract the image id number from the name
        self._sample_id = self._name.split("_")[0]          # Extract the sample id from the name
        self._cells = []                                    # List of cell objects associated with the Image

    def draw_cells(self, col=None, width=2):
        """
        :param col: rbg tuple to describe colour
        :param width: int to describe the width of the line
        :return: image containing annotated cells
        """
        img = cv2.imread(self._path)
        [cell.draw(img, col, width) for cell in self._cells]
        return img

    def add_cell(self, cell):
        """
        :param cell:
        :return:
        """
        self._cells.append(cell)

    def total_cells(self, prediction=None):
        """
        :return:
        """
        if prediction is None:
            return len(self._cells)
        else:
            return len([1 for cell in self._cells if cell.get_prediction() == prediction])

    def add_cells(self, cells):
        """
        :param cells:
        :return:
        """
        for cell in cells:
            self.add_cell(cell)

    def get_cells(self, complete=False):
        """
        :return:
        """
        if complete:
            return [cell for cell in self._cells if cell.is_complete()]
        else:
            return self._cells

    def get_id(self):
        """
        :return:
        """
        return self._id

    def get_sample_id(self):
        """
        :return:
        """
        return self._sample_id

    def get_path(self):
        """
        :return:
        """
        return self._path

    def get_name(self):
        """
        :return:
        """
        return self._name

    def get_type(self):
        """
        :return:
        """
        return self._type

    def get_image(self):
        """
        :return:
        """
        return cv2.imread(self._path)
