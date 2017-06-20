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
        self._training_samples = []
        self._test_samples = []
        self._results = pd.DataFrame()

    def manually_classify_circles(self):
        # Passes the training samples to the cell detector
        # The user can then manually classify each potential cell
        self._cell_detector.manually_classify_circles(self._training_samples)

    def train_cell_detector(self):
        self._cell_detector.train()

    def detect_rbc(self):
        self._cell_detector.detect_rbc(self._test_samples)

    def load_test_samples(self, file):
        # Given an yaml file
        # Initialises an Image object for every image found in the file
        # Attributes each Image object to a relevant Sample object
        print("Loading test samples.")
        if os.path.isfile(file):
            with open(file) as file:
                test_dict = yaml.load(file)
            total_images = 0
            for image_path in test_dict["images"]:
                if os.path.isfile(image_path):                                      # If the path exists
                    sample_id = helpers.get_sample_id(image_path)                   # Extract sample id from sample path
                    sample_index = helpers.get_sample_index(self._test_samples, sample_id)  # Get index of the sample
                    if sample_index != -1:                                          # If the sample already exists
                        self._test_samples[sample_index].add_image(image_path)      # Add the image to the sample
                    else:
                        sample = Sample(sample_id)                                  # Initialise a new sample object
                        sample.add_image(image_path)                                # Add new image to the sample
                        self._test_samples.append(sample)                           # Append sample to list of samples
                    total_images += 1
                else:
                    print("Warning: " + image_path + " is not a file.")
            print("Successfully loaded " + str(len(self._test_samples)) + " sample(s) with "
                  + str(total_images) + " images.")
        else:
            print("Warning: " + str(file) + " is not a file.")

    def load_training_samples(self, file):
        # Given an yaml file
        # Initialises an Image object for every image found in the file
        # Attributes each Image object to a relevant Sample object
        print("Loading training samples...")
        if os.path.isfile(file):
            with open(file) as file:
                test_dict = yaml.load(file)
            total_images = 0
            for image_path in test_dict["images"]:
                if os.path.isfile(image_path):                                      # If the path exists
                    sample_id = helpers.get_sample_id(image_path)                   # Extract sample id from sample path
                    sample_index = helpers.get_sample_index(self._training_samples,
                                                            sample_id)              # Get index of sample in list
                    if sample_index != -1:                                          # If the sample already exists
                        self._training_samples[sample_index].add_image(image_path)  # Add the image to the sample
                    else:
                        sample = Sample(sample_id)                                  # Initialise a new sample object
                        sample.add_image(image_path)                                # Add new image to the sample
                        self._training_samples.append(sample)                       # Append sample to list of samples
                    total_images += 1
                else:
                    print("Warning: " + image_path + " is not a file.")
            print("Successfully loaded " + str(len(self._training_samples))
                  + " sample(s), totalling " + str(total_images) + " image(s).")
        else:
            print("Warning: " + str(file) + " is not a file.")


