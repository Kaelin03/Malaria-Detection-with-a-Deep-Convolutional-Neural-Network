#!/usr/bin/env python3

import os
import yaml
import pandas as pd

from Sample import Sample
from CellDetector import CellDetector
from Classifier import Classifier


class DiagnosisSystem(object):

    def __init__(self, config):
        self._train_samples = self._load_samples(config["train"])           # Automatically load train samples
        self._test_samples = self._load_samples(config["test"])             # Automatically load test samples
        self._cell_detector = CellDetector(config["cell_detector"])         # Initialise the cell detector
        # self._classifier = Classifier(config["classifier"])               # TODO: Initialise classifier
        self._results = pd.DataFrame()                                      # Data frame to store results

    def train(self):
        for sample in self._train_samples:                                  # For every training sample
            for image in sample.get_images():                               # For every image in the sample
                cells = self._cell_detector.run(image)                      # Get cells
                for cell in cells:                                          # For every cell
                    image.add_cell(cell)                                    # Add the cell to the image
        # TODO: Call classifier.train(self._train_samples) to train the CNN

    def test(self):
        pass

    def _load_samples(self, yaml_path):
        print("Loading samples...")
        samples = []                                                        # Initialise a list to store samples
        config_yaml_path = "../config/" + yaml_path                         # yaml will be in the config dir
        if os.path.isfile(config_yaml_path):                                # If the yaml file exists
            with open(config_yaml_path) as file:                            # Open the file
                image_dict = yaml.load(file)                                # Load the yaml file into a dict
            total_images = 0                                                # Count number of images loaded
            for image_path in image_dict["images"]:                         # For each image path given
                image_path = "../samples/" + image_path                     # Images will be in the samples dir
                if os.path.isfile(image_path):                              # Check the image path exists
                    sample_id = self._get_sample_id(image_path)             # Get the sample id from the image path
                    if sample_id:                                           # If the image has a sample id
                        sample_index = self._get_sample_index(samples, sample_id)   # Get the list index to the sample
                        if sample_index != -1:                              # If sample does exists in samples
                            samples[sample_index].add_image(image_path)     # Add image to the sample
                        else:
                            sample = Sample(sample_id)                      # Create new sample object
                            sample.add_image(image_path)                    # Add image to the sample
                            samples.append(sample)                          # Append sample to samples
                        total_images += 1                                   # Add to the number of images loaded
                    else:
                        print("Warning: no sample id found for " + image_path + ".")
                else:
                    print("Warning: " + image_path + " is not a file.")
            print("Successfully loaded " + str(total_images) + " images from "
                  + str(len(samples)) + " samples(s) given by " + yaml_path + ".")
        else:
            print("Warning: " + yaml_path + " is not a file.")
        return samples

    @staticmethod
    def _get_sample_index(samples, sample_id):
        # Given a sample name
        # Returns the index of the sample if it exists
        # Returns -1 if the sample does not exist
        index = -1
        for i, sample in enumerate(samples):
            if sample_id == sample.get_id():
                index = i
                break
        return index

    @staticmethod
    def _get_sample_id(image_path):
        # Given the path to an image
        # Returns the image name from the string
        # Returns False if no sample id is found
        try:
            sample_id = image_path.split("/")[-1].split("_")[-2]
        except IndexError:
            print("Warning: no image name found in " + image_path + ".")
            sample_id = False
        return sample_id

