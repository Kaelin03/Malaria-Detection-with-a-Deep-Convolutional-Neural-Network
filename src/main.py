#!/usr/bin/env python3

import argparse
from DiagnosisSystem import DiagnosisSystem


def main(args):
    system_config = {"cell_detector": args.cell_detector_config,
                     "classifier": args.classifier_config,
                     "manually_classify": args.manually_classify,
                     "train": args.train,
                     "evaluate": args.evaluate,
                     "test": args.test}
    ds = DiagnosisSystem(system_config)
    while True:
        option = input("""What would you like to do?
    1. manually classify
    2. train
    3. evaluate
    4. test
    5. quit\n""")
        if option == "manually classify" or option == "1":
            ds.manually_classify()
        elif option == "train" or option == "2":
            ds.train()
        elif option == "evaluate" or option == "3":
            ds.evalute()
        elif option == "test" or option == "4":
            ds.test()
        elif option == "quit" or option == "5":
            break

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--manually-classify", dest="manually_classify", default="manually_classify.yaml")
    parser.add_argument("--train", default="train.yaml")
    parser.add_argument("--evaluate", default="evaluate.yaml")
    parser.add_argument("--test", default="test.yaml")
    parser.add_argument("--cell-detector-config", dest="cell_detector_config", default="cell_detector.yaml")
    parser.add_argument("--classifier-config", dest="classifier_config", default="classifier.yaml")
    arguments = parser.parse_args()
    main(arguments)
