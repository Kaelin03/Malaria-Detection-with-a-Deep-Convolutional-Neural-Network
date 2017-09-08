#! /usr/bin/env python3

import argparse
from DiagnosisSystem import DiagnosisSystem


def main(args):
    """
    :param args:
    :return:
    """
    ds = DiagnosisSystem(args.config)
    while True:
        option = input("""What would you like to do?
    1. train cnn
    2. load cnn
    3. evaluate cnn
    4. evaluate for all thresholds
    5. load svm
    6. diagnose
    7. quit\n""")
        if option == "train cnn" or option == "1":
            ds.train_cnn()
        elif option == "load cnn" or option == "2":
            ds.load_cnn()
        elif option == "evaluate cnn" or option == "3":
            ds.evaluate_cnn()
        elif option == "4":
            ds.evaluate_for_all_thresholds()
        elif option == "load svm" or option == "5":
            ds.load_svm()
        elif option == "diagnose" or option == "6":
            ds.diagnose()
        elif option == "quit" or option == "7":
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    main(arguments)

