#!/usr/bin/env python3

from Image import Image


class Sample(object):

    def __init__(self, my_id):
        """
        :param my_id:
        """
        self._id = my_id
        self._images = []

    def add_image(self, image_path):
        """
        :param image_path:
        :return:
        """
        self._images.append(Image(image_path))

    def total_cells(self, status=None):
        """
        :return:
        """
        return sum([image.total_cells(status) for image in self._images])

    def get_images(self):
        """
        :return: images
        """
        return self._images

    def get_id(self):
        """
        :return: _id
        """
        return self._id
