#!/usr/bin/env python3

import cv2

class CellDetector(object):

    def __init__(self):
        self._rbc = None
        self._wbc = None

    def find_wbc(self):
        pass

    def find_rbc(self, image_path):


    def run(self):
        self.find_rbc()
        self.find_wbc()
        return {"rbc": self._rbc, "wbc": self._wbc}
