#!/usr/bin/env python3

import argparse
from src.System import System


def main(args):
    system_config = {"cell_detector_config": args.cell_detector_config,
                     "classifier_config": args.classifier_config}
    system = System(system_config)
    system.load_training_samples(args.train)
    system.load_test_samples(args.test)
    # system.manually_classify_circles()
    system.train_cell_detector()
    system.detect_rbc()


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--cell-detector-config", dest="cell_detector_config", default="cell_detector.yaml")
    parser.add_argument("--classifier-config", dest="classifier_config", default="classifier.yaml")
    arguments = parser.parse_args()
    main(arguments)
