# parallel_try.py

# To treat with vector objects.
import numpy as np

# Everything related to the ML models.
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from sklearn.datasets import make_regression
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import LinearRegression

# To make graphics.
import matplotlib.pyplot as plt

# To system info.
import os
import sys

# To paralaellize.
import concurrent.futures

# To hinting.
from typing import List, Dict, Tuple, Any, Union

# To control dates and times.
import time

# Selfmade imports.
from helpers_to_load_subsets_of_data_and_get_results import cargar_datos_para_un_mes_y_obtener_la_tupla
from utils import DataLoader
from modelling import Metricas

# To show progress bars.
from tqdm import tqdm


def timeit_decorator_with_pid(func, verbose: bool = True):
    def wrapper(*args, **kwargs):
        pid = os.getpid()
        start_time = time.monotonic()
        result = func(*args, **kwargs)
        end_time = time.monotonic()
        duration = round(end_time - start_time, 3)
        if verbose:
            print(f"PID: {pid}: Execution time of {func.__name__}: {duration} seconds")
        return result, start_time, end_time, duration
    return wrapper

def info_of_training_and_evaluate(type_of_execution="parallel"):
    def decorator(func):
        def wrapper(*args, **kwargs):
            models, predictions, times, metrics = func(*args, **kwargs)
            
            if type_of_execution == "parallel":
                time = round(max(times), 3)
            if type_of_execution == "serial":
                time = round(sum(times), 3)

            print("\n\n")
            print(f"\n#### Summary of {type_of_execution} execution ####")
            print("--------------------------------------------------------------------------------")
            print(f"\nTime of the parallel: {time}\n")
            for i in range(4):
                print(f"Metrics for model {i} in parallel: {round(metrics[i], 3)}")
            print(f"\nMean of metrics in parallel: {round(np.mean(list(metrics.values())), 3)}\n")
            print("--------------------------------------------------------------------------------")
            return models, predictions, times, metrics
        return wrapper
    return decorator

@timeit_decorator_with_pid
def train_a_single_model(type_of_model: str = "random_forest",
                         tuple_of_inputs: Tuple[np.ndarray, np.ndarray] = (),
                         params_of_model: Dict[str, Any] = {},
                         seed: int = 42
                         ) -> Union[RandomForestRegressor, XGBRegressor, LGBMRegressor, GradientBoostingRegressor]:

    """
    # Function to train one model.
    ---
    
    ## Params:
    ---
    
    - type_of_model : str = "random_forest" 
    - tuple_of_inputs : Tuple[Any] = ()
    - params_of_model : Dict[str, Any] = {}
    - seed : int = 42

    ## Outputs:
    - model : Union[RandomForestRegressor, XGBRegressor, LGBMRegressor, GradientBoostingRegressor]
    ---

    """

    X, y = tuple_of_inputs

    if type_of_model == "random_forest":
        model = RandomForestRegressor(**params_of_model)
    elif type_of_model == "xgboost":
        model = XGBRegressor(**params_of_model)
    elif type_of_model == "lightgbm":
        model = LGBMRegressor(**params_of_model)
    elif type_of_model == "gradient_boosting":
        model = GradientBoostingRegressor(**params_of_model)

    model.fit(X, y)
    return model

@info_of_training_and_evaluate(type_of_execution="parallel")
def parallel(data_of_updrs_prepared_to_models, type_of_model : str = 'random_forest', HYPERPARAMETERS : Dict[str, Any] = {}):

    print("\n\n")
    print("--------------------------------------------------------------------------------")
    print(f"Hyperparameters of the model {type_of_model}:")
    for key, vale in HYPERPARAMETERS[type_of_model].items():
        print(f"Key: {key}, Value: {vale}")
    print("\n")

    # Entrenamiento en paralelo
    models_parallel = {}
    times_parallel = []
    metrics_parallel = {}

    # Create the data tuples to process.
    data_tuples = [(data_of_updrs_prepared_to_models[i][0], data_of_updrs_prepared_to_models[i][3]) for i in range(4)]

    with concurrent.futures.ThreadPoolExecutor(max_workers=4) as executor:

        # Crear un objeto tqdm para la barra de progreso
        future_to_index = {executor.submit(train_a_single_model,
                                        type_of_model=type_of_model,
                                        tuple_of_inputs=data_tuples[i],
                                        params_of_model=HYPERPARAMETERS[type_of_model],
                                        seed=HYPERPARAMETERS[type_of_model]["random_state"]): i for i in range(4)}
        
        for future in tqdm(concurrent.futures.as_completed(future_to_index), total=len(future_to_index), desc="Parallel Training"):
            index = future_to_index[future]
            model, _start_time, _end_time, _duration = future.result()
            models_parallel[index] = model
            times_parallel.append((_duration))

    for i in range(4):
        # Realizar predicciones
        y_pred_parallel = models_parallel[i].predict(data_of_updrs_prepared_to_models[i][2])  # X_test

        # Calcular métricas
        y_true = data_of_updrs_prepared_to_models[i][5]  # y_test
        metrics_parallel[i] = Metricas.smape(y_true, y_pred_parallel)

    print("--------------------------------------------------------------------------------")
    return models_parallel, y_pred_parallel, times_parallel, metrics_parallel

    
@info_of_training_and_evaluate(type_of_execution="serial")
def serial(data_of_updrs_prepared_to_models, type_of_model : str = 'random_forest', HYPERPARAMETERS : Dict[str, Any] = {}):

    print("\n\n")
    print("--------------------------------------------------------------------------------")
    print(f"Hyperparameters of the model {type_of_model}:")
    for key, vale in HYPERPARAMETERS[type_of_model].items():
        print(f"Key: {key}, Value: {vale}")
    print("\n")

    # Entrenamiento en serie
    models_serial = {}
    times_serial = []
    metrics_serial = {}

    # Create the data tuples to process.
    data_tuples = [(data_of_updrs_prepared_to_models[i][0], data_of_updrs_prepared_to_models[i][3]) for i in range(4)]

    for i in tqdm(range(4), desc="Serial Training"):
        model, _start_time, _end_time, _duration = train_a_single_model(type_of_model=type_of_model,
                                                                        tuple_of_inputs=data_tuples[i],
                                                                        params_of_model=HYPERPARAMETERS[type_of_model],
                                                                        seed=HYPERPARAMETERS[type_of_model]["random_state"])
        models_serial[i] = model
        times_serial.append((_duration))

    for i in range(4):
        # Realizar predicciones
        y_pred_serial = models_serial[i].predict(data_of_updrs_prepared_to_models[i][2])  # X_test

        # Calcular métricas
        y_true = data_of_updrs_prepared_to_models[i][5]  # y_test
        metrics_serial[i] = Metricas.smape(y_true, y_pred_serial)

    print("--------------------------------------------------------------------------------")
    return models_serial, y_pred_serial, times_serial, metrics_serial

def plot_feature_importance(model, feature_names, title, max_features=None):
    """
    Plot the feature importances of a model.
    """
    importances = model.feature_importances_
    indices = np.argsort(importances)[::-1]

    # Si se especifica un número máximo de características, limitar la visualización
    if max_features is not None:
        indices = indices[:max_features]

    plt.figure(figsize=(8, 4))
    plt.title(title)
    plt.bar(range(len(indices)), importances[indices], color="skyblue", align="center")
    plt.xticks(range(len(indices)), [model.feature_names_in_[i] for i in indices], rotation=0, size=6)
    plt.xlim([-1, len(indices)])
    plt.ylim([0, 0.6])
    plt.show()


def main(month_to_predict : int = 24, type_of_model : str = 'random_forest', type_of_execution : str = 'parallel'):

    print("\n\n #### New execution of parallel_try.py #### \n\n")
    print('#####################################################')

    HYPERPARAMETERS_THROUGH_SCRIPT = {
        'month_to_predict' : month_to_predict,
        'random_state' : 42,
    }

    HYPERPARAMETERS_OF_MODELS = {
        "random_forest_old" : {
            "n_estimators" : int(50),
            "n_jobs" : 4,
            "random_state" : HYPERPARAMETERS_THROUGH_SCRIPT['random_state'],
            'max_features' : 1.0,
            'bootstrap' : False,
        },
        "random_forest": {
            "n_estimators": 10,  # Un número moderado de árboles
            "max_depth": None,  # Profundidad máxima para cada árbol (sin límite)
            # "min_samples_split": 2,  # Mínimo número de muestras requeridas para dividir un nodo
            # "min_samples_leaf": 1,  # Mínimo número de muestras requeridas en un nodo hoja
            # "max_features": 'sqrt',  # Número de características a considerar en cada división ("sqrt" es una buena regla general)
            "bootstrap": False,  # Desactivar el bootstrap
            "random_state": HYPERPARAMETERS_THROUGH_SCRIPT['random_state'],  # Para resultados reproducibles
            "n_jobs": 4,  # Usar todos los procesadores disponibles,
            # 'criterion' : 'poisson',
        },
        "xgboost" : {
            "booster": 'gbtree',
            "colsample_bylevel": 0.7,
            "colsample_bytree": 0.7,
            "gamma": 0.1,
            "learning_rate": 0.05,
            "max_depth": 6,
            "min_child_weight": 3,
            "n_estimators": 100,
            "reg_alpha": 0.01,
            "reg_lambda": 1,
            "subsample": 0.8,
            "random_state": 42
        },
        "lightgbm" : {
            "n_estimators" : int(50),
            "n_jobs" : 4,
            "random_state" : HYPERPARAMETERS_THROUGH_SCRIPT['random_state']
        },
        "gradient_boosting" : {
            "n_estimators" : int(50),
            "random_state" : HYPERPARAMETERS_THROUGH_SCRIPT['random_state'],
            "n_jobs" : 4
        }
    } 

    data_of_updrs_prepared_to_models = cargar_datos_para_un_mes_y_obtener_la_tupla(mes_a_predecir=HYPERPARAMETERS_THROUGH_SCRIPT["month_to_predict"]) # Here contains X_train, X_val, X_test, y_train, y_val, y_test

    if type_of_execution == "parallel":
        models, predictions, times, metrics = parallel(data_of_updrs_prepared_to_models=data_of_updrs_prepared_to_models, type_of_model=type_of_model, HYPERPARAMETERS=HYPERPARAMETERS_OF_MODELS)
    elif type_of_execution == "serial":
        models, predictions, times, metrics = serial(data_of_updrs_prepared_to_models=data_of_updrs_prepared_to_models, type_of_model=type_of_model, HYPERPARAMETERS=HYPERPARAMETERS_OF_MODELS)
    else:
        raise Exception("The type of execution is not correct. Please, choose between 'parallel' or 'serial'.")
    
    # En la función main, llama a plot_feature_importance con todas las características
    for i, model in models.items():
        print(len(model.feature_names_in_))
        print("\n")
        print(model.feature_names_in_)
        # print(list(zip(data_of_updrs_prepared_to_models[i][0], model.feature_importances_)))
        plot_feature_importance(model, feature_names=data_of_updrs_prepared_to_models[i][0].columns.to_list(), title=f"Model {i} - Feature Importances", max_features=None)


if __name__ == "__main__":

    main(month_to_predict  = 24, type_of_model  = 'random_forest', type_of_execution  = 'parallel')


    