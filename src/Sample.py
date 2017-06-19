#!/usr/bin/env python3


class Sample(object):

    def __init__(self, name):
        self._name = name
        self._rbc = []
        self._wbc = []

    def get_name(self):
        return self._name
