import traceback
from typing import Dict, Tuple
import pandas as pd

from utils import DataLoader
from core import tfm_simple_model_new
from configuration import parse_arguments

def main():
    """Main function of the script."""
    args = parse_arguments()

    print(f"Option Optuna: {args.option_optuna}")
    print(f"Option Parallel: {args.option_parallel}")
    print(f"Seed: {args.seed_value}")
    print(f"Prediction Of: {args.prediction_of}")
    print(f"Type of Model: {args.type_of_model}")
    print(f"Month to make the prediction: {args.month_to_predict}")

    if args.prediction_of:
        dict_of_dfs: Dict[str, pd.DataFrame] = {}
        dict_of_dfs['proteins'], dict_of_dfs['peptides'], dict_of_dfs['clinical'], _ = DataLoader().load_train_data()
        tuple_of_dict_dfs_to_predict_and_month = (dict_of_dfs, args.month_to_predict)

        tfm_simple_model_new(
            type_of_model=args.type_of_model,
            make_predictions_selfmade=args.prediction_of,
            option_parallel=args.option_parallel, 
            option_optuna=args.option_optuna, 
            seed=args.seed_value,
            tuple_of_dict_dfs_to_predict_and_month=tuple_of_dict_dfs_to_predict_and_month)
    else:
        tfm_simple_model_new(
            type_of_model=args.type_of_model,
            option_parallel=args.option_parallel, 
            option_optuna=args.option_optuna, 
            seed=args.seed_value, 
            make_predictions_selfmade=args.prediction_of)

if __name__ == '__main__':
    try:
        main()
    except Exception as e:
        print(f"Exception: {e}")
        traceback.print_exc()
        raise e


