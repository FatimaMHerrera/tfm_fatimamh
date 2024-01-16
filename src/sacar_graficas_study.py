import os
import json
import joblib
import optuna.visualization as ov
import optuna
from optuna.visualization import plot_optimization_history
from optuna.visualization import plot_param_importances
from optuna.visualization import plot_parallel_coordinate
from optuna.visualization import plot_contour
# import matplotlib.pyplot as plt

if __name__ == '__main__':

    HIPERPARAMETROS = {
    'mes_default' : 24,
    'mes_a_predecir' : 84,
    'semilla' : 42,
    'n_jobs' : 16,
    'tipo_de_model' : 'random_forest', 
    'modelos' : ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'],
    'ruta_de_mejores_modelos_joblibs' : os.path.join(os.getcwd(), '..', 'data', 'outputs', 'pruebas', 'mejores_modelos_joblibs'),
    'ruta_de_mejores_modelos_parametros_json' : os.path.join(os.getcwd(), '..', 'data', 'outputs', 'pruebas', 'mejores_modelos_parametros_json'),
    'ruta_study_optuna' : os.path.join(os.getcwd(), '..', 'data', 'outputs', 'pruebas', 'study_optuna'),
    }

    # Cargar los modelos y los parámetros.
    models = {}
    params = {}
    studies = {}

    for i in range(1, 5):
        model_path = os.path.join(HIPERPARAMETROS['ruta_de_mejores_modelos_joblibs'], f"{HIPERPARAMETROS['tipo_de_model']}_model_{i}.joblib")
        params_path = os.path.join(HIPERPARAMETROS['ruta_de_mejores_modelos_parametros_json'], f"{HIPERPARAMETROS['tipo_de_model']}_params_model_{i}.json")

        # Cargar los modelos.
        models[f'model_{i}'] = joblib.load(model_path)

        # Cargar los parámetros.
        with open(params_path, 'r') as file:
            params[f'model_{i}'] = json.load(file)

        # Cargar los estudios de Optuna.
        study_path = os.path.join(HIPERPARAMETROS['ruta_study_optuna'], f"{HIPERPARAMETROS['tipo_de_model']}_study_model_{i}.pkl")
        studies[f'model_{i}'] = joblib.load(study_path)

        # import optuna.visualization as ov
        for i in range(1, 5):
            try:
                optuna.visualization.plot_contour(studies[f'model_{i}']).show(renderer="browser")
            except Exception as e:
                print(e)