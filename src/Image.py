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
        self._cells = []

    def add_cell(self, cell):
        """
        :param cell:
        :return:
        """
        self._cells.append(cell)

    def total_cells(self, status=None):
        """
        :return:
        """
        if status is None:
            return len(self._cells)
        else:
            n = 0
            for cell in self._cells:
                if cell.get_status() == status:
                    n += 1
            return n

    def add_cells(self, cells):
        """
        :param cells:
        :return:
        """
        for cell in cells:
            self.add_cell(cell)

    def draw_cells(self, img, col=(0, 255, 0), width=2):
        """
        :param img:
        :param col:
        :param width:
        :return:
        """
        for cell in self._cells:
            cell.draw(img, col, width)

    def get_cells(self):
        """
        :return:
        """
        return self._cells

    def get_id(self):
        """
        :return:
        """
        return self._id

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
