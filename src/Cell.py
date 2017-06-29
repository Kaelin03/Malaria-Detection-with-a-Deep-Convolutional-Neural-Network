#!/usr/bin/env python3

import cv2


class Cell(object):

    def __init__(self, position, radius):
        self._position = tuple(map(int, position))
        self._radius = int(radius)
        self._status = None

    def draw(self, image, col=(0, 255, 0), width=2):
        cv2.circle(image, self._position, self._radius, col, width)

    def get_position(self):
        # Returns the position of the cell
        return self._position

    def get_radius(self):
        # Returns the radius of the cell
        return self._radius

    def get_status(self):
        # Returns the status of the cell
        return self._status
