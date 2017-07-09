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
    4. save model
    5. load model
    6. plot model
    7. save history
    8. plot history
    9. - plot filters
    10. - diagnose
    11. quit\n""")
        if option == "manually classify" or option == "1":
            ds.manually_classify()
        elif option == "train" or option == "2":
            ds.train()
        elif option == "evaluate" or option == "3":
            ds.evaluate()
        elif option == "save model" or option == "4":
            ds.save_model()
        elif option == "load model" or option == "5":
            ds.load_model()
        elif option == "plot model" or option == "6":
            ds.plot_model()
        elif option == "save history" or option == "7":
            ds.save_history()
        elif option == "plot history" or option == "8":
            ds.plot_history()
        elif option == "plot filters" or option == "9":
            ds.plot_filters()
        elif option == "diagnose" or option == "10":
            ds.diagnose()
        elif option == "quit" or option == "11":
            break
        else:
            print("Invalid selection.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config.yaml")
    arguments = parser.parse_args()
    main(arguments)
