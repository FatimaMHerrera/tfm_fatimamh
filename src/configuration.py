import argparse
import sys
import traceback

class MyParser(argparse.ArgumentParser):
    """Class to pass command line arguments."""

    def __init__(self):
        super().__init__(description="Fatima's TFM main script.")

        # Aquí debes usar 'self' en lugar de 'parser'
        self.add_argument('-tom', '--type_of_model', type=str, default='random_forest',
                          choices=['xgboost', 'random_forest', 'gradient_boosting', 'lightgbm'],
                          help='Model type to use in the script.')
        self.add_argument('-op', '--option_parallel', action='store_true',
                          help='Enable parallel computation.')
        self.add_argument('-no-op', '--no_option_parallel', action='store_false',
                          help='Disable parallel computation.')
        self.set_defaults(option_parallel=False)
        self.add_argument('-oo', '--option_optuna', action='store_true',
                          help='Enable Optuna optimization.')
        self.add_argument('-no-oo', '--no_option_optuna', action='store_false',
                          help='Disable Optuna optimization.')
        self.set_defaults(option_optuna=False)
        self.add_argument('-seed', '--seed_value', type=int, default=42,
                          help='Seed for reproducibility.')
        self.add_argument('-m', '--month_to_predict', type=int, default=24,
                          help='Month for prediction.')
        self.add_argument('-predof', '--prediction_of', action='store_true',
                          help='Make predictions of a dict of dataframes.')
        self.add_argument('-no-predof', '--no_prediction_of', action='store_false',
                          help='Do not make predictions of a dict of dataframes.')
        self.set_defaults(prediction_of=False)

    def error(self, message):
        sys.stderr.write('error: %s\n' % message)
        self.print_help()
        sys.exit(2)

def parse_arguments():
    parser = MyParser()
    return parser.parse_args()