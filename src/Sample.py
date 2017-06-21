#!/usr/bin/env python3

from Image import Image


class Sample(object):

    def __init__(self, my_id):
        self._id = my_id
        self._images = []

    def add_image(self, image_path):
        # Give an image path
        # Initialises new Image object and appends to list of images
        self._images.append(Image(image_path))

    def get_images(self):
        # Returns the list of images
        return self._images

    def get_id(self):
        # Returns the name of the sample
        return self._id
