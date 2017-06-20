#!/usr/bin/env python3

import os
import yaml
import pandas as pd

import src.helpers as helpers
from src.Sample import Sample
from src.CellDetector import CellDetector
from src.Classifier import Classifier


class System(object):

    def __init__(self, config):
        self._cell_detector = CellDetector()
        self._cell_detector.configure(config["cell_detector_config"])
        self._classifier = Classifier()
        self._training_images = []
        self._samples = []
        self._results = pd.DataFrame()

    def manually_classify_circles(self, file):
        #
        #
        if os.path.isfile(file):
            self._load_training_samples(file)
            for image_path in self._training_images:
                self._cell_detector.manually_classify_circles(image_path)
        else:
            print("Warning: " + file + " is not a file.")

    def test(self, file):
        #
        #
        if os.path.isfile(file):
            self._set_test_samples(file)
        else:
            if file is None:
                print("Warning: No test information given")
            else:
                print("Warning: " + file + " is not a file.")

    def _set_test_samples(self, file):
        # Given an yaml file
        # Initialises an Image object for every image found in the file
        # Attributes each Image object to a relevant Sample object
        with open(file) as file:
            test_dict = yaml.load(file)
        for image_path in test_dict["images"]:
            if os.path.isfile(image_path):                                      # If the path exists
                sample_id = helpers.get_sample_id(image_path)                   # Extract sample id from sample path
                sample_index = self._get_sample_index(sample_id)                # Get index of the sample
                if sample_index != -1:                                          # If the sample already exists
                    self._samples[sample_index].add_image(image_path)           # Add the image to the sample
                else:
                    sample = Sample(sample_id)                                  # Initialise a new sample object
                    sample.add_image(image_path)                                # Add new image to the sample
                    self._samples.append(sample)                                # Append sample to list of samples
            else:
                print("Warning: " + image_path + " is not a file.")
        print("Successfully loaded " + str(len(self._samples)) + " sample(s).")

    def _load_training_samples(self, file):
        # Given a yaml file
        # Loads the sample images into a dictionary
        with open(file) as file:
            self._training_images = yaml.load(file)["images"]
        print("Successfully loaded " + str(len(self._training_images)) + " training image(s).")

    def _get_sample_index(self, sample_id):
        # Given a sample name
        # Returns the index of the sample if it exists
        # Returns -1 if the sample does not exist
        index = -1
        for i, sample in enumerate(self._samples):
            if sample_id == sample.get_id():
                index = i
                break
        return index
