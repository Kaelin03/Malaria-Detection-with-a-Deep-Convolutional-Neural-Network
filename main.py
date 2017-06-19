#!/usr/bin/env python3

import argparse
from src.System import System


def main(args):
    system_config = {"cell_detector_config": args.cell_detector_config,
                     "classifier_config": args.classifier_config}
    system = System(system_config)
    system.manually_classify_circles(args.train)
    # system.test(args.test)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--test")
    parser.add_argument("--cell-detector-config", dest="cell_detector_config", default="cell_detector.yaml")
    parser.add_argument("--classifier-config", dest="classifier_config", default="classifier.yaml")
    arguments = parser.parse_args()
    main(arguments)
