#!/usr/bin/env python3

from keras.models import Sequential                             # Class for specifying and training a Neural Network
from keras.models import load_model                             # For loading models
from keras.layers import Conv2D                                 # For convolutional layers
from keras.layers import MaxPooling2D                           # For max pooling layers
from keras.layers import Dense                                  # For fully-connected layers
from keras.layers import Dropout                                # For dropout layers
from keras.layers import Flatten                                # For converting to 1D
from keras.utils import plot_model                              # For graphical representation of model
from keras.utils import np_utils                                # Utilities for one-hot encoding of ground truth values
from keras.optimizers import rmsprop
import matplotlib.pyplot as plt

import numpy as np
import pandas as pd
import random
import copy
import math
import cv2
import os

import helpers


class NeuralNetwork(object):

    def __init__(self, yaml_path):
        """
        Initialises all Class attributes and updates them with values given in the config file
        :param yaml_path: path to the configuration file
        """
        self._yaml_path = "../config/" + yaml_path
        self._name = None
        self._history = None
        self._kernel_size = None
        self._batch_size = None
        self._train_batches = None
        self._evaluate_batches = None
        self._augment = None
        self._channel_shift = None
        self._drop_prob = None
        self._conv_depth = None
        self._hidden_size = None
        self._learn_rate = None
        self._pool_size = None
        self._activation = None
        self._model = None
        self._epochs = None
        self._init = None
        self._threshold = None
        self._image_shape = None
        self._classes = None
        self._train_data = None
        self._evaluate_data = None
        self.update_config()

    def update_config(self):
        config = helpers.load_yaml(self._yaml_path)
        self._kernel_size = [(value, value) for value in config["neural_network"]["kernel_size"]]
        self._epochs = config["neural_network"]["epochs"]
        self._train_batches = config["train"]["batches"]
        self._evaluate_batches = config["evaluate"]["batches"]
        self._batch_size = config["neural_network"]["batch_size"]
        self._drop_prob = config["neural_network"]["drop_prob"]
        self._conv_depth = config["neural_network"]["conv_depth"]
        self._hidden_size = config["neural_network"]["hidden_size"]
        self._learn_rate = config["neural_network"]["learn_rate"]
        self._activation = config["neural_network"]["activation"]
        self._augment = config["neural_network"]["augment"]
        self._threshold = config["evaluate"]["threshold"]
        self._channel_shift = config["neural_network"]["channel_shift"]
        self._init = config["neural_network"]["init"]
        self._pool_size = [(value, value) for value in config["neural_network"]["pool_size"]]
        self._train_data = "../" + config["train"]["data"]
        self._evaluate_data = config["evaluate"]["data"]
        self._classes = config["classes"]
        self._image_shape = (config["images"]["height"],
                             config["images"]["width"],
                             config["images"]["depth"])
        self.reset_history()

    def train(self):
        self.update_config()
        if self._model is None:
            self.compile_model()
        self._name = input("Enter a name for the model:\n")
        self.draw_model()
        image_paths = self.get_image_paths()
        X, Y = self.get_data(image_paths)
        try:
            max_val_acc = 0.75
            for epoch in range(self._epochs):
                X, Y = helpers.shuffle_arrays(X, Y)
                for batch in range(self._train_batches):
                    e = epoch + batch / self._train_batches
                    print("Epoch " + str(e) + "/" + str(self._epochs))
                    x, y = self.get_batch(X, Y, batch, self._train_batches)
                    if self._augment:
                        x = self.augment(x, self._channel_shift)
                    x = self.normalise(x)
                    y = np_utils.to_categorical(y, len(self._classes))
                    self.store_history(self._model.fit(x, y,
                                                       batch_size=self._batch_size,
                                                       epochs=1,
                                                       verbose=1,
                                                       validation_split=0.1), e)
                    self.update_plot()
                    if self._history["val_acc"][-1] > max_val_acc:
                        max_val_acc = self._history["val_acc"][-1]
                        self.save_model()
        except KeyboardInterrupt:
            print("\nWarning: training interrupted!")
        while True:
            option = input("Would you like to save the model now? y/n")
            if option == "y":
                self.save_model()
                break
            if option == "n":
                break

    def evaluate(self):
        self.update_config()
        if self._model is not None:
            total = {"tp": 0, "tn": 0, "fp": 0, "fn": 0, "n": 0, "p": 0, "t": 0, "f": 0}
            for directory in self._evaluate_data:
                summary = self.set_summary()
                image_paths = self.get_image_paths2("../" + directory)
                X, Y = self.get_data(image_paths)
                image_paths = [path for paths in image_paths for path in paths]
                print("Evaluating dataset")
                with open("../results/cell_data/" + directory.split("/")[-1] + ".csv", "w") as file:
                    file.write("cell path,status,confidence\n")
                count = 0
                helpers.progress_bar(0, self._evaluate_batches,
                                     prefix="Progress", suffix="Complete", length=30, fill="=")
                for batch in range(self._evaluate_batches):
                    x, y = self.get_batch(X, Y, batch, self._evaluate_batches)
                    # x = self.normalise(x)
                    x /= 255
                    predictions = self._model.predict(x,
                                                      batch_size=self._batch_size,
                                                      verbose=0)
                    for label, prediction in zip(y, predictions):
                        status = self.get_status(label, self.get_prediction(prediction, self._threshold))
                        summary[status] += 1
                        with open("../results/cell_data/" + directory.split("/")[-1] + ".csv", "a") as file:
                            file.write(",".join([image_paths[count], status, str(prediction[1])]) + "\n")
                        count += 1
                    helpers.progress_bar(batch + 1, self._evaluate_batches,
                                         prefix="Progress", suffix="Complete", length=30, fill="=")
                summary["sample"] = directory.split("/")[-1]
                try:
                    summary["sensitivity"] = summary["tp"] / (summary["tp"] + summary["fn"])
                except ZeroDivisionError:
                    print("Warning: no positive cases.")
                try:
                    summary["specificity"] = summary["tn"] / (summary["tn"] + summary["fp"])
                except ZeroDivisionError:
                    print("Warning: no negative cases.")
                print(summary)
                total["tp"] += summary["tp"]
                total["fp"] += summary["fp"]
                total["tn"] += summary["tn"]
                total["fn"] += summary["fn"]
            total["n"] = total["tn"] + total["fp"]
            total["p"] = total["tp"] + total["fn"]
            total["t"] = total["tp"] + total["tn"]
            total["f"] = total["fp"] + total["fn"]
            print(total)
        else:
            print("Warning: no model configured.")

    def evaluate_for_all_thresholds(self, steps=21):
        self.update_config()
        if self._model is not None:
            data = {"Threshold": [],
                    "TN": [], "TP": [], "FN": [], "FP": [],
                    "N": [], "P": [], "T": [], "F": [],
                    "TNR": [], "FNR": [], "TPR": [], "FPR": [],
                    "PPV": [], "NPV": [],
                    "LRp": [], "LRn": [],
                    "ACC": [], "F1": [],
                    "MCC": [], "Informedness": [], "Markedness": []}
            for threshold in np.linspace(0, 1, steps):
                print("Threshold: " + str(threshold))
                summary = {"tn": 0, "tp": 0, "fn": 0, "fp": 0}
                for directory in self._evaluate_data:
                    image_paths = self.get_image_paths2("../" + directory)
                    X, Y = self.get_data(image_paths)
                    print("Evaluating dataset")
                    helpers.progress_bar(0, self._evaluate_batches,
                                         prefix="Progress", suffix="Complete", length=30, fill="=")
                    for batch in range(self._evaluate_batches):
                        x, y = self.get_batch(X, Y, batch, self._evaluate_batches)
                        # x = self.normalise(x)
                        x /= 255
                        predictions = self._model.predict(x,
                                                          batch_size=self._batch_size,
                                                          verbose=0)
                        for label, prediction in zip(y, predictions):
                            status = self.get_status(label, self.get_prediction(prediction, threshold))
                            summary[status] += 1
                        helpers.progress_bar(batch + 1, self._evaluate_batches,
                                             prefix="Progress", suffix="Complete", length=30, fill="=")
                data["Threshold"].append(threshold)
                data = self.fill_data(data, summary)
            data = pd.DataFrame.from_dict(data)
            data.to_csv("../data.csv", sep=",")
        else:
            print("Warning: model not configured.")

    def predict(self, x):
        x /= 255
        predictions = self._model.predict(x,
                                          batch_size=self._batch_size,
                                          verbose=1)
        return predictions

    @staticmethod
    def fill_data(data, summary):
        tp = summary["tp"]
        tn = summary["tn"]
        fn = summary["fn"]
        fp = summary["fp"]
        data["TN"].append(tn)
        data["TP"].append(tp)
        data["FN"].append(fn)
        data["FP"].append(fp)
        data["N"].append(tn + fp)
        data["P"].append(tp + fn)
        data["T"].append(tn + tp)
        data["F"].append(fn + fp)
        try:
            data["TNR"].append(tn / (tn + fp))
        except ZeroDivisionError:
            data["TNR"].append("nan")
        try:
            data["FNR"].append(fn / (fn + tp))
        except ZeroDivisionError:
            data["FNR"].append("nan")
        try:
            data["TPR"].append(tp / (tp + fn))
        except ZeroDivisionError:
            data["FPR"].append("nan")
        try:
            data["FPR"].append(fp / (fp + tn))
        except ZeroDivisionError:
            data["FPR"].append("nan")
        try:
            data["PPV"].append(tp / (tp + fp))
        except ZeroDivisionError:
            data["PPV"].append("nan")
        try:
            data["NPV"].append(tn / (tn + fn))
        except ZeroDivisionError:
            data["NPV"].append("nan")
        try:
            data["LRp"].append((tp / (tp + fn)) / (1 - (tn / (tn + fp))))
        except ZeroDivisionError:
            data["LRp"].append("nan")
        try:
            data["LRn"].append((1 - (tp / (tp / fn))) / (tn / (tn + fp)))
        except ZeroDivisionError:
            data["LRn"].append("nan")
        try:
            data["ACC"].append((tp + tn) / (tp + fp + tn + fn))
        except ZeroDivisionError:
            data["ACC"].append("nan")
        try:
            data["F1"].append(2 * tp / (2 * tp + fp + fn))
        except ZeroDivisionError:
            data["F1"].append("nan")
        try:
            data["MCC"].append((tp * tn - fp * fn) / math.sqrt((tp + fp) * (tp + fn) * (fn + fp) * (tn + fn)))
        except ZeroDivisionError:
            data["MCC"].append("nan")
        try:
            data["Informedness"].append(tp / (tp + fn) + tn / (tn + fp) - 1)
        except ZeroDivisionError:
            data["Informedness"].append("nan")
        try:
            data["Markedness"].append(tp / (tp + fp) + tn / (tn + fn) - 1)
        except ZeroDivisionError:
            data["Markedness"].append("nan")
        return data

    @staticmethod
    def get_prediction(prediction, threshold):
        if prediction[1] > threshold:
            return 1
        else:
            return 0

    @staticmethod
    def set_summary():
        return {"sample": None,
                "tn": 0, "tp": 0, "fn": 0, "fp": 0,
                "sensitivity": None,
                "specificity": None}

    @staticmethod
    def get_status(label, prediction):
        if label == prediction and label == 0:
            return "tn"
        elif label == prediction and label == 1:
            return "tp"
        elif label != prediction and label == 0:
            return "fp"
        elif label != prediction and label == 1:
            return "fn"
        else:
            return "N/A"

    def save_model(self):
        directory = "../models/" + self._name
        file_name = self._name + ".h5"
        if not os.path.isdir(directory):
            os.makedirs(directory)
        self._model.save(directory + "/" + file_name)
        print("Model saved to " + directory + "/" + file_name + ".")

    def load_model(self):
        model_name = input("Enter the model name:\n")
        directory = "../models/" + model_name.split(".")[0]
        if model_name[-3] != ".h5":
            model_name += ".h5"
        try:
            self._model = load_model(directory + "/" + model_name)
            print("Successfully loaded " + model_name + ".")
        except OSError:
            print("Warning: " + directory + "/" + model_name + " not found.")

    def save_plot(self):
        directory = "../models/" + self._name
        file_name = "history.png"
        if not os.path.isdir(directory):
            os.makedirs(directory)
        plt.savefig(directory + "/" + file_name)
        print("Plot saved to " + directory + "/" + file_name)
        plt.show()

    def update_plot(self):
        plt.clf()
        loss = self._history["loss"]
        val_loss = self._history["val_loss"]
        acc = self._history["acc"]
        val_acc = self._history["val_acc"]
        epoch = self._history["epoch"]
        plt.figure(1)
        plt.subplot(121)
        plt.plot(epoch, acc, "red", label="Training Accuracy")
        plt.plot(epoch, val_acc, "blue", label="Validation accuracy")
        plt.legend(loc=0, frameon=False)
        plt.ylabel("Accuracy")
        plt.xlabel("Epoch")
        plt.ylim([0, 1])
        plt.xlim([0, self._epochs])
        plt.grid()
        plt.subplot(122)
        plt.plot(epoch, loss, "red", label="Training loss")
        plt.plot(epoch, val_loss, "blue", label="Validation loss")
        plt.legend(loc=0, frameon=False)
        plt.ylabel("Loss")
        plt.xlabel("Epoch")
        plt.ylim([0, 1])
        plt.xlim([0, self._epochs])
        plt.grid()
        plt.pause(0.25)

    def draw_model(self, ):
        directory = "../models/" + self._name
        file_name = self._name + ".png"
        if not os.path.isdir(directory):
            os.makedirs(directory)
        try:
            plot_model(self._model,
                       to_file=directory + "/" + file_name,
                       show_shapes=True,
                       show_layer_names=False)
            print("Diagram saved to " + directory + "/" + file_name + ".")
        except:
            print("Warning: unable to draw model.")

    def save_history(self):
        directory = "../models/" + self._name
        file_name = "history.csv"
        if not os.path.isdir(directory):
            os.makedirs(directory)
        file = open(directory + "/" + file_name, "w")
        keys = [key for key in self._history]
        epochs = len(self._history[keys[0]])
        file.write(",".join(keys) + "\n")
        for i in range(epochs):
            data = [self._history[key][i] for key in keys]
            data = ",".join(list(map(str, data)))
            file.write(data + "\n")
        file.close()
        print("History saved to " + directory + "/" + file_name + ".")

    def store_history(self, episode, epoch):
        self._history["acc"].append(episode.history["acc"][0])
        self._history["val_acc"].append(episode.history["val_acc"][0])
        self._history["loss"].append(episode.history["loss"][0])
        self._history["val_loss"].append(episode.history["val_loss"][0])
        self._history["epoch"].append(epoch)

    def reset_history(self):
        self._history = {"epoch": [],
                         "acc": [],
                         "loss": [],
                         "val_acc": [],
                         "val_loss": []}

    @staticmethod
    def normalise(x):
        return (x - 127.5) / 127.5

    @staticmethod
    def augment(x, channel_shift):
        print("Performing channel shifts...")
        helpers.progress_bar(0, x.shape[0], prefix="Progress", suffix="Complete", length=30, fill="=")
        for row in range(x.shape[0]):
            for ch in range(x.shape[3]):
                x[row] += random.randrange(channel_shift[0], channel_shift[1])
            helpers.progress_bar(row + 1, x.shape[0], prefix="Progress", suffix="Complete", length=30, fill="=")
        x[x > 255] = 255
        x[x < 0] = 0
        return x

    @staticmethod
    def get_batch(X, Y, batch, n_batches):
        n = Y.shape[0]
        lower = batch * n // n_batches
        upper = (batch + 1) * n // n_batches
        x = copy.deepcopy(X[lower:upper])
        y = copy.deepcopy(Y[lower:upper])
        x = x.astype(np.float32)
        return x, y

    def get_data(self, image_paths):
        n = sum([len(paths) for paths in image_paths])
        x = np.empty((n,
                      self._image_shape[0],
                      self._image_shape[1],
                      self._image_shape[2]), dtype=np.uint8)
        y = np.empty((n, 1), dtype=np.uint8)
        row = 0
        for label, paths in enumerate(image_paths):
            print("Loading images from class " + str(label))
            l = len(paths)
            helpers.progress_bar(0, l, prefix="Progress", suffix="Complete", length=30, fill="=")
            for i, path in enumerate(paths):
                y[row] = label
                x[row] = cv2.imread(path)
                row += 1
                helpers.progress_bar(i + 1, l, prefix="Progress", suffix="Complete", length=30, fill="=")
        return x, y

    def get_image_paths(self):
        image_paths = [os.listdir(self._train_data + "/" + label) for label in self._classes]
        for i, label in enumerate(self._classes):
            image_paths[i] = [self._train_data + "/" + label + "/" + image for image in image_paths[i]]
            # image_paths[i] = image_paths[i][0:80000]
        return image_paths

    def get_image_paths2(self, directory):
        image_paths = [helpers.get_file_names(directory + "/" + label) for label in self._classes]
        return image_paths

    def compile_model(self):
        self._model = Sequential()
        # Conv2D, Conv2D, Pool, Dropout
        self._model.add(Conv2D(self._conv_depth[0], self._kernel_size[0],
                               input_shape=self._image_shape,
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        self._model.add(Conv2D(self._conv_depth[1], self._kernel_size[1],
                               input_shape=self._image_shape,
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        self._model.add(Conv2D(self._conv_depth[2], self._kernel_size[2],
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        self._model.add(MaxPooling2D(self._pool_size[0]))
        self._model.add(Dropout(self._drop_prob[0], seed=0))

        # Conv2D, Conv2D, Pool, Dropout
        self._model.add(Conv2D(self._conv_depth[2], self._kernel_size[3],
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        self._model.add(Conv2D(self._conv_depth[3], self._kernel_size[4],
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        self._model.add(Conv2D(self._conv_depth[0], self._kernel_size[5],
                               input_shape=self._image_shape,
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        self._model.add(MaxPooling2D(self._pool_size[1]))
        self._model.add(Dropout(self._drop_prob[1], seed=1))

        # Conv2D, Conv2D, Pool, Dropout
        self._model.add(Conv2D(self._conv_depth[4], self._kernel_size[6],
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        self._model.add(Conv2D(self._conv_depth[5], self._kernel_size[7],
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        self._model.add(Conv2D(self._conv_depth[0], self._kernel_size[8],
                               input_shape=self._image_shape,
                               padding="valid",
                               activation=self._activation,
                               kernel_initializer=self._init))
        # self._model.add(MaxPooling2D(self._pool_size[2]))
        self._model.add(Dropout(self._drop_prob[2], seed=2))

        # Flatten
        self._model.add(Flatten())

        # Dense, Dropout
        self._model.add(Dense(self._hidden_size[0],
                              activation=self._activation,
                              kernel_initializer=self._init))
        self._model.add(Dropout(self._drop_prob[3], seed=4))
        # Dense, Dropout
        self._model.add(Dense(self._hidden_size[1],
                              activation=self._activation,
                              kernel_initializer=self._init))
        self._model.add(Dropout(self._drop_prob[4], seed=5))

        # Dense, Dropout

        self._model.add(Dense(self._hidden_size[2],
                              activation=self._activation,
                              kernel_initializer=self._init))
        self._model.add(Dropout(self._drop_prob[5], seed=6))

        # Dense - softmax
        self._model.add(Dense(len(self._classes),
                        activation="softmax",
                        kernel_initializer=self._init))

        opt = rmsprop(lr=self._learn_rate, decay=1e-6)

        self._model.compile(loss="categorical_crossentropy",
                            optimizer=opt,
                            metrics=["accuracy"])
