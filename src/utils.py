# utils.py

# To use the operating system functionalities.
import os

# To use the project's root path.
import pathlib

# To use dataframes.
import pandas as pd

# to use numpy arrays.
import numpy as np

# To use decorators.
from functools import wraps

# To measure execution time.
import time

class ProjectRootPaths:

    """Class that stores the paths of the project's root, src and data folders."""

    def __init__(self):
        self.root = self.find_project_root()
        self.src_path = self.root / "src"
        self.data_path = self.root / "data"

    def find_project_root(self):
        # Start from the current working directory.
        current_path = pathlib.Path(os.getcwd())

        # Upwards in the directory tree.
        for parent in current_path.parents:
            if (parent / "src").exists() and (parent / "data").exists():
                return parent
        raise FileNotFoundError("Cannot find project root.")

    def get_src_path(self):
        return self.src_path

    def get_data_path(self):
        return self.data_path

class PathStorage(ProjectRootPaths):

    """Class that stores the paths of the whole project"""

    def __init__(self):
        super().__init__()  # Initialize the attributes of the base class.

        # The base class attributes are used to define the following attributes.

        # General directories.
        self.DIR_BASE_CODE = os.path.join(self.src_path)
        self.DIR_BASE_DATA = os.path.join(self.data_path)
        self.DIR_INPUTS = os.path.join(self.DIR_BASE_DATA, 'inputs')
        self.DIR_OUTPUTS = os.path.join(self.DIR_BASE_DATA, 'outputs')

        # Inputs > csv files.
        self.DIR_TRAIN_CLINICAL = os.path.join(self.DIR_INPUTS, 'train_clinical_data.csv')
        self.DIR_TRAIN_PROTEINS = os.path.join(self.DIR_INPUTS, 'train_proteins.csv')
        self.DIR_TRAIN_PEPTIDES = os.path.join(self.DIR_INPUTS, 'train_peptides.csv')
        self.SUPLEMENTAL_CLINICAL_DATA = os.path.join(self.DIR_INPUTS, 'supplemental_clinical_data.csv')
        self.DIR_TEST_CLINICAL = os.path.join(self.DIR_INPUTS, 'test.csv')
        self.DIR_TEST_PROTEINS = os.path.join(self.DIR_INPUTS, 'test_proteins.csv')
        self.DIR_TEST_PEPTIDES = os.path.join(self.DIR_INPUTS, 'test_peptides.csv')

        # Outputs > graphics.
        self.DIR_IMAGES = os.path.join(self.DIR_OUTPUTS, 'graphics')

        # Outputs > models > joblibs.
        self.DIR_MODELS_JOBLIBS = os.path.join(self.DIR_OUTPUTS, 'best_models_joblibs')

        # Outputs > models > jsons.
        self.DIR_PARAMETERS_JSONS = os.path.join(self.DIR_OUTPUTS, 'best_parameters_jsons')

        # Outputs > models > metrics.
        self.DIR_METRICS = os.path.join(self.DIR_OUTPUTS, 'metrics')

        # Outputs > models > study_optuna.
        self.DIR_STUDY_OPTUNA = os.path.join(self.DIR_OUTPUTS, 'study_optuna')

class DataLoader(PathStorage):

    def __init__(self):
        super().__init__()

    def load_train_data(self):
        proteins = pd.read_csv(self.DIR_TRAIN_PROTEINS)
        peptides = pd.read_csv(self.DIR_TRAIN_PEPTIDES)
        clinical = pd.read_csv(self.DIR_TRAIN_CLINICAL)
        supplement = pd.read_csv(self.SUPLEMENTAL_CLINICAL_DATA)
        return proteins, peptides, clinical, supplement

    def load_test_data(self):
        proteins = pd.read_csv(self.DIR_TEST_PROTEINS)
        peptides = pd.read_csv(self.DIR_TEST_PEPTIDES)
        clinical = pd.read_csv(self.DIR_TEST_CLINICAL)
        return proteins, peptides, clinical
    

def reduce_mem_usage(df):
    start_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage of dataframe is {:.2f} MB'.format(start_mem))

    for col in df.columns:
        col_type = df[col].dtype

        if col_type != object:
            c_min = df[col].min()
            c_max = df[col].max()
            if str(col_type)[:3] == 'int':
                if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                    df[col] = df[col].astype(np.int8)
                elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                    df[col] = df[col].astype(np.int16)
                elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                    df[col] = df[col].astype(np.int32)
                elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                    df[col] = df[col].astype(np.int64)  
            else:
                if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                    df[col] = df[col].astype(np.float16)
                elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                    df[col] = df[col].astype(np.float32)
                else:
                    df[col] = df[col].astype(np.float64)
        else:
            df[col] = df[col].astype('category')

    end_mem = df.memory_usage().sum() / 1024**2
    print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
    print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

    return df

class DataFrameOptimizer:

    __slots__ = ['df', 'verbose']
    
    def __init__(self, df : pd.DataFrame, verbose=False) -> None:
        self.df = df
        self.verbose = verbose

    def __call__(self) -> pd.DataFrame:
        return self.reduce_mem_usage()

    def reduce_mem_usage(self):

        start_mem = self.df.memory_usage().sum() / 1024**2

        for col in self.df.columns:
            col_type = self.df[col].dtype

            if col_type != object:
                c_min = self.df[col].min()
                c_max = self.df[col].max()
                if str(col_type)[:3] == 'int':
                    if c_min > np.iinfo(np.int8).min and c_max < np.iinfo(np.int8).max:
                        self.df[col] = self.df[col].astype(np.int8)
                    elif c_min > np.iinfo(np.int16).min and c_max < np.iinfo(np.int16).max:
                        self.df[col] = self.df[col].astype(np.int16)
                    elif c_min > np.iinfo(np.int32).min and c_max < np.iinfo(np.int32).max:
                        self.df[col] = self.df[col].astype(np.int32)
                    elif c_min > np.iinfo(np.int64).min and c_max < np.iinfo(np.int64).max:
                        self.df[col] = self.df[col].astype(np.int64)  
                else:
                    if c_min > np.finfo(np.float16).min and c_max < np.finfo(np.float16).max:
                        self.df[col] = self.df[col].astype(np.float16)
                    elif c_min > np.finfo(np.float32).min and c_max < np.finfo(np.float32).max:
                        self.df[col] = self.df[col].astype(np.float32)
                    else:
                        self.df[col] = self.df[col].astype(np.float64)
            else:
                self.df[col] = self.df[col].astype('category')

        end_mem = self.df.memory_usage().sum() / 1024**2
        
        if self.verbose:
            print('Memory usage of dataframe before optimizacion is {:.2f} MB'.format(start_mem))
            print('Memory usage after optimization is: {:.2f} MB'.format(end_mem))
            print('Decreased by {:.1f}%'.format(100 * (start_mem - end_mem) / start_mem))

        return self.df
    
def timing_decorator(func):
    @wraps(func)
    def wrapper(*args, **kwargs):
        start_time = time.monotonic()
        result = func(*args, **kwargs)
        end_time = time.monotonic()
        print(f"Execution time of '{func.__name__}': {end_time - start_time:.2f} seconds\n\n")
        return result
    return wrapper