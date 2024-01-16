import os
import json
import numpy as np
import matplotlib.pyplot as plt
import joblib

from modelling import Metricas
from sklearn.model_selection import (learning_curve, KFold)
from sklearn.ensemble import (GradientBoostingRegressor, RandomForestRegressor)
from sklearn.metrics import (mean_squared_error, mean_absolute_error, r2_score)
import xgboost as xgb
import matplotlib.pyplot as plt
import lightgbm as lgb

from typing import (List, Dict, Tuple, Optional, Union, Any, Callable)
import pandas as pd

from edafecomp import initialization_eda_fe
from utils import (timing_decorator, DataLoader, PathStorage)

def plottear(predicciones : np.ndarray,
             y_test : np.ndarray,
             type_of_model: str = 'random_forest', 
             updrs_number : int = '1', 
             verbose : bool = False,
             ruta_outputs : str = PathStorage().DIR_IMAGES
             ) -> None:

    # Gráfica de Dispersión de Predicciones vs Valores Reales
    plt.scatter(y_test, predicciones)
    plt.xlabel('Real Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs. Real Values for the updrs {updrs_number} model')
    plt.savefig(f'predicciones_vs_valores_reales_{type_of_model}_{updrs_number}.png')

    if verbose == True:
        plt.show()

    # Histograma de los Errores de Predicción
    errores = predicciones - y_test
    plt.hist(errores, bins=25)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of errors in the prediction for the updrs {updrs_number} model')
    
    if verbose == True:
        plt.show()
    else:
        ruta_to_save = os.path.join(ruta_outputs, f'predicciones_vs_valores_reales_{type_of_model}_{updrs_number}.png')
        plt.savefig(ruta_to_save)
        plt.close()  # Cierra la figura para liberar memoria


def plottear(predicciones : np.ndarray,
             y_test : np.ndarray,
             type_of_model: str = 'random_forest', 
             updrs_number : int = '1', 
             verbose : bool = False,
             ruta_outputs : str = PathStorage().DIR_IMAGES
             ) -> None:

    # Gráfica de Dispersión de Predicciones vs Valores Reales
    plt.scatter(y_test, predicciones)
    plt.xlabel('Real Values')
    plt.ylabel('Predictions')
    plt.title(f'Predictions vs. Real Values for the updrs {updrs_number} model')

    if verbose == True:
        plt.show()
    else:
        ruta_to_save = os.path.join(ruta_outputs, f'predicciones_vs_valores_reales_{type_of_model}_{updrs_number}.png')
        plt.savefig(ruta_to_save)
        plt.close()

    # Histograma de los Errores de Predicción
    errores = predicciones - y_test
    plt.hist(errores, bins=25)
    plt.xlabel('Prediction Error')
    plt.ylabel('Frequency')
    plt.title(f'Histogram of errors in the prediction for the updrs {updrs_number} model')
    
    if verbose == True:
        plt.show()
    else:
        ruta_to_save = os.path.join(ruta_outputs, f'predicciones_vs_valores_reales_{type_of_model}_{updrs_number}.png')
        plt.savefig(ruta_to_save)
        plt.close()  # Cierra la figura para liberar memoria


# Función para generar la curva de aprendizaje
def plot_learning_curve(estimator, 
                        title, 
                        X, 
                        y, 
                        cv=None, 
                        n_jobs=None, 
                        train_sizes=np.linspace(.1, 1.0, 5),
                        verbose : bool = False,
                        ruta_outputs = PathStorage().DIR_IMAGES,
                        updrs_number : int = '1',
                        type_of_model : str = 'random_forest'
                        ):
    plt.figure()
    plt.title(title)
    plt.xlabel("Training examples")
    plt.ylabel("Score")

    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=n_jobs, train_sizes=train_sizes)
    train_scores_mean = np.mean(train_scores, axis=1)
    train_scores_std = np.std(train_scores, axis=1)
    test_scores_mean = np.mean(test_scores, axis=1)
    test_scores_std = np.std(test_scores, axis=1)

    plt.grid()

    plt.fill_between(train_sizes, train_scores_mean - train_scores_std,
                     train_scores_mean + train_scores_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_scores_mean - test_scores_std,
                     test_scores_mean + test_scores_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_scores_mean, 'o-', color="r",
             label="Training score")
    plt.plot(train_sizes, test_scores_mean, 'o-', color="g",
             label="Cross-validation score")

    plt.legend(loc="best")

    if verbose == True:
        plt.show()
    else:
        ruta_to_save = os.path.join(ruta_outputs, f'learning_curve_{type_of_model}_model_{updrs_number}.png')
        plt.savefig(ruta_to_save)
        plt.close()  # Cierra la figura para liberar memoria

# Función para graficar los residuales frente a los valores ajustados
def plot_residuals_vs_fitted(y_test, 
                             predictions,
                             ruta_outputs = PathStorage().DIR_IMAGES,
                             verbose : bool = False,
                             type_of_model : str = 'random_forest',
                             updrs_number : int = '1'
                             ):
    
    """Fuction to plot the residuals vs. fitted values."""
    
    residuals = y_test - predictions
    plt.scatter(predictions, residuals)
    plt.axhline(y=0, color='r', linestyle='--')
    plt.xlabel('Fitted values')
    plt.ylabel('Residuals')
    plt.title('Residuals vs. Fitted')

    if verbose == True:
        plt.show()
    else:
        ruta_to_save = os.path.join(ruta_outputs, f'residuals_vs_fitted_{type_of_model}_model_{updrs_number}.png')
        plt.savefig(ruta_to_save)
        plt.close()  # Cierra la figura para liberar memoria


def graficar_resultados(predictions, data_sets, ruta_outputs = PathStorage().DIR_IMAGES, verbose = False):
    for i, (pred, data_set) in enumerate(zip(predictions, data_sets), 1):
        _, _, X_test, _, _, y_test = data_set
        plottear(pred, y_test, type_of_model = 'random_forest', updrs_number=str(i), ruta_outputs=ruta_outputs, verbose=verbose)