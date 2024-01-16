# For the reception of the arguments when the script is called.
import argparse

# For the use of warnings.
import warnings
warnings.simplefilter(action='ignore', category=(FutureWarning, UserWarning))

# To use DataFrames.
import pandas as pd
pd.options.mode.chained_assignment = None  # default='warn'
pd.set_option('display.max_columns', None)

# To treat mathematical objects.
import numpy as np 

# To get online information.
import requests

# To get random samples or numbers.
import random as rnd

# For the typing hints.
from typing import Dict, List, Any, Tuple

# To interact with the operative system.
import os
from concurrent.futures import (ProcessPoolExecutor, as_completed)

# To interact with properties of the functions.
from functools import wraps

# To get information about the errors.
import traceback

# To the graphics.
import matplotlib.pyplot as plt
import seaborn as sns

# To train, preprocess and evaluate models.

import xgboost as xgb
import lightgbm as lgb

from sklearn.model_selection import (train_test_split, KFold, learning_curve)
from sklearn.preprocessing import (MinMaxScaler, StandardScaler)
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor)
from sklearn.impute import SimpleImputer
from sklearn.metrics import (mean_squared_error, mean_absolute_error)
from sklearn.linear_model import LinearRegression
from sklearn.exceptions import DataConversionWarning
from sklearn.compose import ColumnTransformer

# Para optimizar hiperparámetros.
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_contour

# Para guardar y cargar modelos.
import joblib

# Para guardar y cargar parámetros.
import json

# Personal imports.
from utils import (DataLoader, DataFrameOptimizer, reduce_mem_usage, timing_decorator, PathStorage)
from bioinfo import UnitProtInfo

from fe import (FeatureEngineeringNew, full_FeatureEngineeringNew)
from eda import (EdaNew, full_EdaNew)
from modelling import (DataPreparationToModelNew, Metricas)

from Bio import BiopythonWarning
warnings.simplefilter('ignore', BiopythonWarning)

import time

from modelling import ModelTrainEvaluate
from modelling import train_and_save_models
from edafecomp import initialization_eda_fe
from optimization import run_optuna_optimization, early_stopping_callback



@timing_decorator
def tfm_simple_model_new(if_sample: bool = False, 
                         sample_size: int = 10, 
                         type_of_model: str = 'random_forest', 
                         option_parallel: bool = False, 
                         option_optuna: bool = False, 
                         seed: int = 42, 
                         make_predictions_selfmade: bool = False, 
                         tuple_of_dict_dfs_to_predict_and_month: Tuple[Dict[str, pd.DataFrame], int] = ()
                         ):

    model_path = os.path.join(PathStorage().DIR_MODELS_JOBLIBS, f'{type_of_model}_model.joblib')
    params_path = os.path.join(PathStorage().DIR_PARAMETERS_JSONS, f'{type_of_model}_optuna_params.json')

    if os.path.exists(model_path) and os.path.exists(params_path):
        print(f"Modelo y parámetros existentes encontrados para {type_of_model}.")

        if make_predictions_selfmade:

            # Cargar modelo
            model = joblib.load(model_path)

            # Cargar parámetros (si es necesario)
            with open(params_path, 'r') as json_file:
                model_params = json.load(json_file)

            # Preparar datos para la predicción
            data_to_models_prepared = initialization_eda_fe(make_predictions_selfmade=make_predictions_selfmade, 
                                                            tuple_of_dict_dfs_to_predict_and_month=tuple_of_dict_dfs_to_predict_and_month)

            # Realizar predicciones y evaluar
            for data_set in data_to_models_prepared:
                # Desempaquetar el conjunto de datos
                X_train, _X_val, X_test, y_train, _y_val, y_test = data_set

                # Realizar predicciones
                y_pred = model.predict(X_test)

                # Evaluar y mostrar resultados
                metric_results = ModelTrainEvaluate.calculate_metrics(y_test, y_pred)
                print(f"Resultados de la predicción para el conjunto de datos: {metric_results}")

        else:
            print("All is already computed and ready to make predictions.")

    else: #TODO (Realizar tests aqui importante.)


        data_to_models_prepared = initialization_eda_fe(
            make_predictions_selfmade=make_predictions_selfmade, 
            tuple_of_dict_dfs_to_predict_and_month=tuple_of_dict_dfs_to_predict_and_month,
            if_sample=if_sample, 
            sample_size=sample_size, 
            seed=seed, 
            include_all_updrs_in_train=False
        )
        
        if option_optuna:
            run_optuna_optimization(data_to_models_prepared, type_of_model, option_parallel, seed)
        else:
            train_and_save_models(data_to_models_prepared, type_of_model, option_parallel, seed)


def tfm_simple_model_new_web(type_of_model, model, tuple_of_dict_dfs_to_predict_and_month, seed=42):

    
    # Preparar los datos para la predicción
    # Asume que dict_of_dfs ya contiene los DataFrames necesarios
    # y month_to_predict es el mes para el cual se realiza la predicción

    # Preparar datos para la predicción
    data_to_models_prepared = initialization_eda_fe(make_predictions_selfmade=True, 
                                                    tuple_of_dict_dfs_to_predict_and_month=tuple_of_dict_dfs_to_predict_and_month)

    # Realizar predicciones y evaluar
    for data_set in data_to_models_prepared:
        # Desempaquetar el conjunto de datos
        X_train, _X_val, X_test, y_train, _y_val, y_test = data_set

        # Realizar predicciones
        y_pred = model.predict(X_test)

        # Evaluar y mostrar resultados
        metric_results = ModelTrainEvaluate.calculate_metrics(y_test, y_pred)

        return metric_results