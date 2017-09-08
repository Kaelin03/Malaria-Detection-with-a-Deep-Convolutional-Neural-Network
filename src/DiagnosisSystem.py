#! /usr/bin/env python3

from NeuralNetwork import NeuralNetwork
from CellDetector import CellDetector
from Sample import Sample

import helpers
import numpy as np
import cv2
import os


class DiagnosisSystem(object):

    def __init__(self, yaml_path):
        """
        :param yaml_path:
        """
        self._yaml_path = "../config/" + yaml_path
        self._neural_network = NeuralNetwork(yaml_path)
        self._cell_detector = CellDetector(yaml_path)

    def train_cnn(self):
        self._neural_network.train()
        self._neural_network.save_history()
        self._neural_network.save_plot()

    def load_cnn(self):
        self._neural_network.load_model()

    def evaluate_cnn(self):
        self._neural_network.evaluate()

    def evaluate_for_all_thresholds(self):
        self._neural_network.evaluate_for_all_thresholds()

    def load_svm(self):
        self._cell_detector.load_model()

    def diagnose(self):
        sample_paths = helpers.load_yaml(self._yaml_path)["diagnose"]["samples"]
        samples = self.load_samples(sample_paths)
        for sample in samples:
            print("Diagnosing sample " + sample.get_id() + "...")
            for image in sample.get_images():
                if image.get_id() == "01":
                    self._cell_detector.run(image)
                    x = self.cells_to_array(image.get_cells())
                    predictions = self._neural_network.predict(x)
                    for prediction, cell in zip(*(predictions, image.get_cells())):
                        cell.set_prediction(np.argmax(prediction))
                        cell.set_confidence(np.max(prediction))


                    directory = "../results/cell_detector4/" + sample.get_id()
                    if not os.path.isdir(directory):
                        os.makedirs(directory)
                    cv2.imwrite(directory + "/" + image.get_name() + ".jpg",
                                image.draw_cells())

    def load_samples(self, directories):
        files = []
        for directory in directories:
            files += helpers.get_file_names("../" + directory)
        image_paths = [file for file in files if file[-4:] == ".jpg" or file[-4:] == ".png"]
        samples = []
        for path in image_paths:
            sample_id = path.split("/")[-1].split("_")[0]
            sample_index = self.get_sample_index(samples, sample_id)
            if sample_index == -1:
                sample = Sample(sample_id)
                sample.add_image(path)
                samples.append(sample)
            else:
                samples[sample_index].add_image(path)
        print("Samples found:")
        [print("\t" + sample.get_id()) for sample in samples]
        return samples

    def cells_to_array(self, cells):
        """
        :param cells:
        :return:
        """
        image = helpers.load_yaml(self._yaml_path)["images"]
        image_shape = (image["height"], image["width"], image["depth"])
        x = np.empty((len(cells),
                      image_shape[0],
                      image_shape[1],
                      image_shape[2]), dtype=np.float32)
        for row, cell in zip(*(x, cells)):
            cell_image = cell.get_image(dy=image_shape[0],
                                        dx=image_shape[1])
            width, height, depth = cell_image.shape
            if cell_image.shape != (80, 80, 3):
                print("cell wrong size!!")
            row[0:width, 0:height, 0:depth] = cell_image
        return x

    @staticmethod
    def get_sample_index(samples, sample_id):
        """
        :param samples: a list of Sample objects
        :param sample_id: the id of a sample
        :return: index number of the sample in the list
        """
        index = -1                                                                  # -1 if sample id does not exist
        for i, sample in enumerate(samples):                                        # For each sample
            if sample_id == sample.get_id():                                        # If the sample id matches
                index = i                                                           # Note the sample index
                break                                                               # Break
        return index