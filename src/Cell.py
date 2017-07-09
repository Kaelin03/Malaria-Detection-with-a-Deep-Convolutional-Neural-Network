#!/usr/bin/env python3

import cv2


class Cell(object):

    def __init__(self, position, radius):
        self._position = tuple(map(int, position))
        self._radius = int(radius)
        self._status = None
        
    def get_image(self, image, dx=0, dy=0):
        """
        Crop the image with the cell in the centre
        If dx and dy are not give, the crop will be to the cell radius
        :param image: image to crop
        :param dx: width to crop the image to
        :param dy: height to crop the image to
        :return: cropped image
        """
        if dx <= 0:                                                                 # If dx <= 0
            dx = self._radius * 2                                                   # Crop to cell radius
        if dy <= 0:                                                                 # If dy <= 0
            dy = self._radius * 2                                                   # Crop to cell radius
        height, width, _ = image.shape                                              # Get size of original image
        x1 = int(max(self._position[0] - dx / 2, 0))                                # Calculate bounding box values
        y1 = int(max(self._position[1] - dy / 2, 0))
        x2 = int(min(self._position[0] + dx / 2, width))
        y2 = int(min(self._position[1] + dy / 2, height))
        return image[y1:y2, x1:x2, :]

    def draw(self, image, col=(0, 255, 0), width=2):
        """
        Draws a circle around the cell on a given image
        :param image: image on which to draw
        :param col: colour of the line
        :param width: width of the line
        """
        cv2.circle(image, self._position, self._radius, col, width)

    def get_position(self):
        """
        :return: position of the cell
        """
        return self._position

    def get_radius(self):
        """
        :return: radius of the cell
        """
        return self._radius

    def get_status(self):
        """
        :return: status of the cell
        """
        return self._status
