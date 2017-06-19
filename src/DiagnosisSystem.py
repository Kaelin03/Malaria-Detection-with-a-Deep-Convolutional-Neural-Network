#!/usr/bin/env python3

import os
import pandas as pd

import Helpers as helpers
from Sample import Sample
from Detector import Detector
from Classifier import Classifier


class DiagnosisSystem(object):

    def __init__(self):
        self._detector = Detector()
        self._classifier = Classifier()
        self._train_samples = []
        self._test_samples = []
        self._results = pd.DataFrame()

    def _get_train_samples(self, info_file):
        with open(info_file) as file:
            for image_path in file:
                image_path = helpers.remove_spaces(image_path)
                image_path = helpers.move_comment(image_path)
                if image_path:
                    if os.path.isfile(image_path):
                        sample_name = helpers.get_sample_name(image_path)
                    else:
                        print("Warning: " + image_path + " is not a file")

    def _get_test_samples(self, info_file):
        pass

    def train(self, train_info):
        if os.path.isfile(train_info):
            self._get_train_samples(train_info)
        else:
            if train_info is not None:
                print("Warning: " + train_info + "is not a file")
            else:
                print("Warning: No training information given")

    def test(self, test_info):
        if os.path.isfile(test_info):
            self._get_test_samples(test_info)
        else:
            if test_info is not None:
                print("Warning: " + test_info + "is not a file")
            else:
                print("Warning: No test information given")
