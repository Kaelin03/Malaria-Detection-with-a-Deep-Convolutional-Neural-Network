#!/usr/bin/env python3

import argparse
from DiagnosisSystem import DiagnosisSystem


def main(args):
    system_config = {"cell_detector": args.cell_detector_config,
                     "classifier": args.classifier_config,
                     "test": args.test,
                     "train": args.train}
    ds = DiagnosisSystem(system_config)
    while True:
        option = input("""What would you like to do?
    - train
    - test
    - quit\n""")
        if option == "train":
            ds.train()
        elif option == "test":
            ds.test()
        elif option == "quit":
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train", default=None)
    parser.add_argument("--test", default=None)
    parser.add_argument("--cell-detector-config", dest="cell_detector_config", default="cell_detector.yaml")
    parser.add_argument("--classifier-config", dest="classifier_config", default="classifier.yaml")
    arguments = parser.parse_args()
    main(arguments)
