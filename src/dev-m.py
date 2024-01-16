# m.py
"""
This will be the main file of the project.
"""

import traceback
import sys
from typing import Dict, Tuple, List, Any, Union, Optional
from configuration import parse_arguments


def main():
    args = parse_arguments()

    print(f"Option Optuna: {args.option_optuna}")
    print(f"Option Parallel: {args.option_parallel}")
    print(f"Seed: {args.seed_value}")
    print(f"Prediction Of: {args.prediction_of}")
    print(f"Type of Model: {args.type_of_model}")
    print(f"Month to make the prediction: {args.month_to_predict}")

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        raise e