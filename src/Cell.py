#!/usr/bin/env python3

import cv2


class Cell(object):

    def __init__(self, position, radius):
        """
        :param position:
        :param radius:
        """
        self._position = tuple(map(int, position))
        self._radius = int(radius)
        self._path = None
        self._sample_id = None
        self._image_id = None
        self._id = None
        self._prediction = None
        self._confidence = None
        self._label = None

    def set_id(self, path, nb):
        """
        :return:
        """
        self._path = path
        self._id = path.split("/")[-1].split(".")[0] + "_" + str(nb)
        self._sample_id = self._id.split("_")[0]
        self._image_id = self._id.split("_")[1]

    def get_image(self, dx=0, dy=0):
        """
        Crop the image with the cell in the centre
        If dx and dy are not give, the crop will be to the cell radius
        :param dx: width to crop the image to
        :param dy: height to crop the image to
        :return: cropped image
        """
        image = cv2.imread(self._path)
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

    def get_image_id(self):
        """
        :return:
        """
        return self._image_id

    def draw(self, image, col=None, width=2):
        """
        Draws a circle around the cell on a given image
        :param image: image on which to draw
        :param col: colour of the line
        :param width: width of the line
        """
        if col is None:
            if self._prediction == 0:
                col = (0, 255, 0)
            elif self._prediction == 1:
                col = (0, 0, 255)
            else:
                col = (0, 0, 0)
        # cv2.circle(image, self._position, self._radius, col, width)
        p1 = self._position[0] - 40, self._position[1] - 40
        p2 = self._position[0] + 40, self._position[1] + 40
        cv2.rectangle(image, p1, p2, col, width)

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

    def get_prediction(self):
        """
        :return: status of the cell
        """
        return self._prediction

    def set_prediction(self, prediction):
        """
        :param prediction:
        :return:
        """
        self._prediction = prediction

    def get_label(self):
        """
        :return:
        """
        return self._label

    def set_label(self, label):
        """
        :param label:
        :return:
        """
        self._label = label

    def set_confidence(self, confidence):
        """
        :param confidence:
        :return:
        """
        self._confidence = confidence

    def get_confidence(self):
        """
        :return:
        """
        return self._confidence
