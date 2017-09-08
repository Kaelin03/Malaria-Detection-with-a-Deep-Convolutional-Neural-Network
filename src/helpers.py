#!/usr/bin/env python3

import os
import yaml
import math
import numpy as np


def progress_bar(iteration, total, prefix="", suffix="", decimals=1, length=100, fill="█"):
    try:
        percent = ("{0:." + str(decimals) + "f}").format(100 * (iteration / total))
        filled_length = int(length * iteration / total)
        bar = fill * filled_length + "-" * (length - filled_length)
        print("\r%s |%s| %s%% %s" % (prefix, bar, percent, suffix), end="\r")
        if iteration == total:
            print()
    except ZeroDivisionError:
        print("None.")


def check_ext(file_name, ext):
    """
    :param file_name:
    :param ext:
    :return:
    """
    if file_name.split(".")[-1] != ext:
        file_name += "." + ext
    return file_name


def pythag(a, b):
    """
    :param a: a length
    :param b: a length
    :return: pythagorean distance, c
    """
    c = math.sqrt(a ** 2 + b ** 2)
    return c


def load_yaml(yaml_path):
    """
    :param yaml_path: path to a yaml file
    :return: dictionary containing the contents of the yaml fle
    """
    yaml_dict = {}                                                              # Initialise yaml_dict
    if os.path.isfile(yaml_path):                                               # If the file exists
        with open(yaml_path) as file:                                           # Open the file
            yaml_dict = yaml.load(file)                                         # Load the file
    else:
        print("Warning: " + yaml_path + " not found.")
    return yaml_dict


def shuffle_arrays(a, b):
    """
    :param a:
    :param b:
    :return:
    """
    state = np.random.get_state()
    np.random.shuffle(a)
    np.random.set_state(state)
    np.random.shuffle(b)
    return a, b


def get_file_names(directory):
    file_names = []
    path = directory
    if os.path.isdir(directory):
        for item in os.listdir(directory):
            item = "/".join([path, item])
            if os.path.isdir(item):
                file_names += get_file_names(item)
            else:
                file_names.append(item)
    else:
        print("Warning: " + directory + " not found.")
    return file_names

