#!/usr/bin/env python3

import src.helpers as helpers


class Image(object):

    def __init__(self, path):
        self._path = path
        self._name = path.split("/")[-1].split(".")[-2]
        self._type = path.split("/")[-1].split(".")[-1]
        self._id = helpers.get_image_id(path)
        self._rbc = []
        self._wbc = []

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
