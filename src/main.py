#!/usr/bin/env python3

import argparse
from DiagnosisSystem import DiagnosisSystem


def main(args):
    ds = DiagnosisSystem(args.config)
    while True:
        option = input("""What would you like to do?
    1. manually classify
    2. train
    3. save model
    4. load model
    5. plot model
    6. evaluate
    7. diagnose
    8. quit\n""")
        if option == "manually classify" or option == "1":
            ds.manually_classify()
        elif option == "train" or option == "2":
            ds.train()
        elif option == "save model" or option == "3":
            ds.save_model()
        elif option == "load model" or option == "4":
            ds.load_model()
        elif option == "plot model" or option == "5":
            ds.plot_model()
        elif option == "evaluate" or option == "6":
            ds.evalute()
        elif option == "diagnose" or option == "7":
            ds.diagnose()
        elif option == "quit" or option == "8":
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    main(arguments)
