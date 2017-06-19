#!/usr/bin/env python3

import src.helpers as helpers


class Image(object):

    def __init__(self, path):
        self._path = path
        self._id = helpers.get_image_id(path)
        self._rbc = []
        self._wbc = []

    def get_id(self):
        # Returns the name of the sample
        return self._id
