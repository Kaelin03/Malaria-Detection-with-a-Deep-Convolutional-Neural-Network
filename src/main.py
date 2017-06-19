#!/usr/bin/env python3

import argparse
from DiagnosisSystem import DiagnosisSystem

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--train")
    parser.add_argument("--test")
    args = parser.parse_args()
    diagnosis_system = DiagnosisSystem()
    diagnosis_system.tain(args.train, args.test)
