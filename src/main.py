#!/usr/bin/env python3

import argparse
from DiagnosisSystem import DiagnosisSystem


def main(args):
    ds = DiagnosisSystem(args.config)
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
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    main(arguments)
