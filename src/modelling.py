#######################################################################################
import numpy as np
import pandas as pd

import xgboost as xgb
import lightgbm as lgb
from sklearn.metrics import mean_absolute_error, mean_squared_error
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.model_selection import train_test_split, KFold
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor

from typing import Dict, Tuple, Set, List

import os
import json
import traceback
import joblib

from typing import List, Tuple, Optional, Dict, Any
from concurrent.futures import ProcessPoolExecutor, as_completed
import optuna
from functools import wraps

from utils import DataLoader
#######################################################################################


class Metricas:
    
    @staticmethod
    def smape(y_true : float, y_pred : float):
        smap = np.zeros(len(y_true))
        
        num = np.abs(y_true - y_pred)
        dem = ((np.abs(y_true) + np.abs(y_pred)) / 2)
        
        pos_ind = (y_true!=0) | (y_pred!=0)
        smap[pos_ind] = num[pos_ind] / dem[pos_ind]
        
        return 100 * np.mean(smap)

    @staticmethod
    def calculate_mean_of_metrics(metrics : List[float]):
        """Calculate the mean of a list of metrics."""
        return np.mean(metrics)
    
    @staticmethod
    def rmse(y_true : float, y_pred : float):
        """Calculate the Root Mean Square Error."""
        return np.sqrt(mean_squared_error(y_true, y_pred))

class DataPreparationToModelNew:

    """This class prepare the data to face the models. \nUsage of the class example:\n \
    data_preparation = DataPreparationToModel(train_results, [f'updrs_{i+1}' for i in range(len(train_results))])
    normalized_train_val_test_sets = data_preparation.process_data()\
    """

    __slots__ = ['data_frames', 'target_columns', 'propotion_of_splitting', 'random_state']

    def __init__(self, data_frames: List[Tuple[pd.DataFrame]], target_columns: List[str], propotion_of_splitting: float = 0.2, random_state: int = 42):
        self.data_frames = data_frames
        self.target_columns = target_columns
        self.propotion_of_splitting = propotion_of_splitting
        self.random_state = random_state

    def __call__(self):
       return self.process_data()

    def split_data(self, df : pd.DataFrame, target_column : List[str]) -> tuple:

        test_size = self.propotion_of_splitting
        val_size = self.propotion_of_splitting / (1 - test_size)  # To maintain the proportion.

        X = df.drop(target_column, axis=1)
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=self.random_state)
        X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=val_size, random_state=self.random_state)

        return X_train, X_val, X_test, y_train, y_val, y_test

    def _normalize_and_split_data(self, df : pd.DataFrame, target_column : List[str]) -> tuple:

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df, target_column)
        
        scaler = MinMaxScaler()
        
        X_train_norm = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns)
        X_val_norm = pd.DataFrame(scaler.transform(X_val), columns=X_val.columns)
        X_test_norm = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns)
        
        return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test

    def normalize_and_split_data(self, df : pd.DataFrame, target_column : List[str]) -> tuple:

        X_train, X_val, X_test, y_train, y_val, y_test = self.split_data(df, target_column)
        
        # Identificar las columnas que no deben ser escaladas
        strings_a_excluir = {'mean', 'median', 'min', 'max', 'std', 'var'}
        columnas_sin_cambios = [col for col in df.columns if len(col.split('_')) > 2 if col.split('_')[2] in strings_a_excluir]
        columnas_para_escalar = [col for col in df.columns if col not in columnas_sin_cambios and col not in target_column]

        preprocesador = ColumnTransformer(
            transformers=[
                ('escalar', StandardScaler(), columnas_para_escalar),
                ('sin_cambio', 'passthrough', columnas_sin_cambios)
            ]
        )

        # Aplicar el preprocesador a los conjuntos de entrenamiento, validación y prueba
        X_train_norm = pd.DataFrame(preprocesador.fit_transform(X_train), columns=columnas_para_escalar + columnas_sin_cambios)
        X_val_norm = pd.DataFrame(preprocesador.transform(X_val), columns=columnas_para_escalar + columnas_sin_cambios)
        X_test_norm = pd.DataFrame(preprocesador.transform(X_test), columns=columnas_para_escalar + columnas_sin_cambios)
        
        return X_train_norm, X_val_norm, X_test_norm, y_train, y_val, y_test

    def process_data(self) -> list:

        processed_data = []
        
        for df, target_column in zip(self.data_frames, self.target_columns):
            processed_data.append(self._normalize_and_split_data(df, target_column))

        return processed_data
    
class ModelTrainEvaluate:

    def __init__(
            self,
            option_parallel: bool = False,
            type_of_model: str = 'random_forest',
            dict_of_metrics_in_use: Dict[str, bool] = {"smape": True, "rmse": True},
            model_params: Dict[str, Any] = None,
            seed: int = 42
            ) -> None:
        
        self.type_of_model = type_of_model
        self.option_parallel = option_parallel
        self.SEED = seed
        self.dict_of_metrics_in_use = dict_of_metrics_in_use  
        self.dict_of_metris_results = {}
        self.results = {f'{type_of_model}': {'smape': [], 'rmse': []}}
        self.models = []

        self.DICT_OF_PARAMETERS_OF_MODELS = {self.type_of_model: model_params} if model_params else {}


    @staticmethod
    def calculate_metrics(y_true, y_pred):

        smape = Metricas.smape(y_true, y_pred)
        rmse = np.sqrt(mean_squared_error(y_true, y_pred))
        mae = mean_absolute_error(y_true, y_pred)

        return {'smape' : smape, 'rmse': rmse, 'mae': mae}
    
    def train_and_save_individual_models(self, data_to_models_prepared, save_path, model_index):
        # Entrenar el modelo para el conjunto de datos actual
        model = self.train_model(data_to_models_prepared[0])  # Asumiendo que data_to_models_prepared contiene solo un conjunto de datos
        self.models.append(model)

        # Guardar el modelo
        model_file_path = os.path.join(save_path, f'{self.type_of_model}_model_{model_index}.joblib')
        joblib.dump(model, model_file_path)
        print(f"Modelo para UPDRS {model_index} guardado en {model_file_path}")
    
    def train_model(
                self,
                data_set, # The input.
                ) -> Dict[str, float]:
    
        """This function tries to train the model and then evaluate it.\
        The steps are:\
        1. We unpack the different arrays from the data_set that is receipt as input.\
        2. Then we take into account the parallel configuration if it proceeds.
        """

        # if self.option_parallel:
        print(f"Ejecutando en proceso con PID: {os.getpid()}")
        
        # The dataset reception and unpacking it.
        X_train, _X_val, X_test, y_train, _y_val, y_test = data_set

        if self.type_of_model == 'xgboost':
            model = xgb.XGBRegressor(random_state=self.SEED, **self.DICT_OF_PARAMETERS_OF_MODELS['xgboost'])
        elif self.type_of_model == 'random_forest':
            model = RandomForestRegressor(random_state=self.SEED, **self.DICT_OF_PARAMETERS_OF_MODELS['random_forest'])
        elif self.type_of_model == 'gradient_boosting':
            model = GradientBoostingRegressor(random_state=self.SEED, **self.DICT_OF_PARAMETERS_OF_MODELS['gradient_boosting'])
        elif self.type_of_model == 'lightgbm':
            model = lgb.LGBMRegressor(random_state=self.SEED, **self.DICT_OF_PARAMETERS_OF_MODELS['lightgbm'])

        # The model will be fitted.
        model.fit(X_train, y_train)

        return model
    
    def make_predictions(self, model, X_test, y_test):

        # Then we make a prediction on the test data.
        y_pred = model.predict(X_test)

        # After the prediction we evaluate it using the metrics (only the metrics in use).
        for metric in self.dict_of_metrics_in_use:
            if self.dict_of_metrics_in_use[metric]:
                if metric == 'smape':
                    self.dict_of_metris_results['smape'] = Metricas.smape(y_test, y_pred)
                elif metric == 'rmse':
                    self.dict_of_metris_results['rmse'] = np.sqrt(mean_squared_error(y_test, y_pred))
                elif metric == 'mae':
                    self.dict_of_metrics_in_use['mae'] = mean_absolute_error(y_test, y_pred)

        return self.dict_of_metris_results
    

    @staticmethod
    def print_results_decorator(func):
        @wraps(func)
        def wrapper(*args, **kwargs):
            # Llamando a la función original y obteniendo la instancia de ModelTrainEvaluate
            model_train_evaluate = args[0]
            func(*args, **kwargs)

            print("\n\n" + 100*"#")
            for model_name, metrics in model_train_evaluate.results.items():
                print(f"\nResultados para {model_name}:")
                print(35*"#")
                for metric, values in metrics.items():
                    if values:
                        print(f"\nValores de {metric.upper()} en cada submodelo:\n")
                        print(35*"-" + "\n")
                        for i in range(len(values)):
                            print(f"sub-modelo {i+1} -> {values[i]}")
                        media_metric = sum(values) / len(values)
                        print(f"\nMedia de {metric.upper()}: {media_metric:.2f}")
                        print(35*"-" + "\n")
                print(35*"#")
        return wrapper

    @print_results_decorator
    def obtaining_the_results(self, data_to_models_prepared):
            if self.option_parallel:
                self._execute_in_parallel(data_to_models_prepared)
            else:
                self._execute_sequentially(data_to_models_prepared)
                
    def _execute_in_parallel(self, data_to_models_prepared): # TODO (Now it seems well sorted)
        with ProcessPoolExecutor(max_workers=4) as executor:
            futures = {}
            for index, data_set in enumerate(data_to_models_prepared):
                future = executor.submit(self._train_and_predict, data_set)
                futures[future] = index  # Guardar el índice de la tarea

            results = [None] * len(data_to_models_prepared)  # Lista para almacenar los resultados en orden

            for future in as_completed(futures):
                result = future.result()
                index = futures[future]  # Obtener el índice correspondiente
                results[index] = result  # Almacenar el resultado en la posición correcta

            # Almacenar los resultados en el orden recibido
            for result in results:
                self._store_results(result)

    def _execute_sequentially(self, data_to_models_prepared):
        for data_set in data_to_models_prepared:
            metric_results = self._train_and_predict(data_set)
            self._store_results(metric_results)

    def _train_and_predict(self, data_set):
        
        # Entrenar el modelo
        model = self.train_model(data_set)

        # Desempaquetar el conjunto de prueba de nuevo pasra poder hacer predicciones.
        X_train, _X_val, X_test, y_train, _y_val, y_test = data_set

        # Hacer predicciones y obtener métricas
        metric_results = self.make_predictions(model, X_test, y_test)
        return metric_results, model

    def _store_results(self, result):
        metric_results, model = result  # Desempaquetar métricas y modelo

        # Almacenar los resultados de las métricas
        for metric, value in metric_results.items():
            self.results[self.type_of_model][metric].append(value)

        # Almacenar el modelo
        self.models.append(model)

    def save_best_models(self, save_path):
        """
        Save the best model and its parameters to the specified directory.
        """
        # Asumiendo que 'results' es un atributo de tu clase que contiene los resultados del modelo
        best_model = self._select_best_model()  # Implementa esta función según tus criterios
        model_name = best_model
        file_path = os.path.join(save_path, f'{model_name}.joblib')

        # Guardar el modelo
        joblib.dump(best_model, file_path)
        print(f"Modelo guardado en {file_path}")

        return best_model

    def _select_best_model(self): # TODO (Maybe argmax)
        """
        Implementa la lógica para seleccionar el mejor modelo.
        Esto es solo un ejemplo. Debes ajustarlo según tus propios criterios.
        """
        best_score = float('inf')
        best_model = None
        for model, metrics in self.results.items():
            # Calcula el promedio de RMSE para cada modelo
            avg_rmse = sum(metrics['rmse']) / len(metrics['rmse'])
            if avg_rmse < best_score:
                best_score = avg_rmse
                best_model = model
        return best_model
    

def train_and_save_models(data_to_models_prepared, type_of_model, option_parallel, seed, save_results=True):

    PARAMETERS_TO_MODELS = {
        'random_forest': {
            "n_estimators": 156, 
            "max_depth": 28, 
            "min_samples_split": 20, 
            "min_samples_leaf": 20, 
            "max_features": "log2", 
            "max_leaf_nodes": 985
        }
    }

    # Ruta del archivo de parámetros
    params_path = os.path.join(DataLoader.DIR_MODELS, f'{type_of_model}_optuna_params.json')

    for i, data_set in enumerate(data_to_models_prepared):
        # Construye la ruta del archivo de parámetros para cada submodelo
        params_file_path = os.path.join(DataLoader.DIR_MODELS, f'{type_of_model}_optuna_params_{i+1}.json')
        
        # Verificar si existe el archivo de parámetros para el submodelo actual
        if os.path.exists(params_file_path):
            with open(params_file_path, 'r') as json_file:
                model_params = json.load(json_file)
        else:
            # Utilizar parámetros predeterminados si no existen parámetros optimizados
            model_params = PARAMETERS_TO_MODELS.get(type_of_model, {})

        # Instanciar ModelTrainEvaluate
        model_train_evaluate = ModelTrainEvaluate(
            type_of_model=type_of_model,
            model_params=model_params,
            option_parallel=option_parallel,
            seed=seed
        )

        # Entrenar y guardar modelos individuales
        model_train_evaluate.train_and_save_individual_models([data_set], DataLoader.DIR_MODELS, i + 1)

# Aquí puedes llamar a la función train_and_save_models con los parámetros necesarios
