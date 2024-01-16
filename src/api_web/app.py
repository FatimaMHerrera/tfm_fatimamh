from flask import Flask, request, render_template
import joblib
import json
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import DataLoader, PathStorage
# from helpers_to_load_subsets_of_data_and_get_results import carga_modelos_y_params
from edafecomp import initialization_eda_fe
from modelling import ModelTrainEvaluate
import pandas as pd

# For the typing hints.
from typing import (Dict, List, Any, Tuple)

from core import tfm_simple_model_new_web

from sklearn.ensemble import (RandomForestRegressor, GradientBoostingRegressor)
from xgboost import XGBRegressor
from lightgbm import LGBMRegressor
from modelling import Metricas

app = Flask(__name__)

path_to_models = PathStorage().DIR_MODELS_JOBLIBS
path_to_params = PathStorage().DIR_PARAMETERS_JSONS

# Verificar si la ruta existe
if os.path.exists(path_to_models):
    # Verificar si la ruta es un directorio
    if os.path.isdir(path_to_models):
        print("El directorio existe.")
    else:
        print("La ruta existe, pero no es un directorio.")
else:
    print("El directorio no existe.")

# Verificar si la ruta existe
if os.path.exists(path_to_params):
    # Verificar si la ruta es un directorio
    if os.path.isdir(path_to_params):
        print("El directorio existe.")
    else:
        print("La ruta existe, pero no es un directorio.")
else:
    print("El directorio no existe.")


models_available = ["random_forest", "xgboost", "lightgbm", "gradient_boosting"]
    
# @app.route('/')
# def index():
#     return render_template('index.html')  # Asegúrate de tener un archivo HTML para el frontend

@app.route('/')
def index():
    lista_de_meses_posibles = [0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 84]
    return render_template('index.html', meses=lista_de_meses_posibles)

@app.route('/predict', methods=['POST'])
def predict():
    try:
        if request.method == 'POST':

            # Se selecciona un modelo.
            modelo_seleccionado = request.form['model']

            # Se selecciona un mes.
            month_to_predict = int(request.form['month_to_predict'])

            # Verificar que el mes esta en esta lista y sino que de un error.
            lista_de_meses_posibles = [0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 84]

            if month_to_predict not in lista_de_meses_posibles:
                return 'You can only take one of this months: 0, 3, 6, 9, 12, 18, 24, 30, 36, 42, 48, 54, 60, 72, 84', 400

            # Cogemos los csv cargados por el usuario.
            clinical = request.files['clinical']
            proteins = request.files['proteins']
            peptides = request.files['peptides']

            # Hay que proporcionar los tres archivos o sino dar un error.
            if not clinical or not proteins or not peptides:
                return 'No se han proporcionado todos los archivos', 400

            # Los encerramos en un diccionario para tenerlos trackeados.
            dict_of_dfs = {
                'clinical': pd.read_csv(clinical),
                'proteins': pd.read_csv(proteins),
                'peptides': pd.read_csv(peptides)
            }

            # Ponemos los modelos y los parametros en una lista.
            list_of_models = [joblib.load(os.path.join(path_to_models, f'{modelo_seleccionado}_model_{i}.joblib')) for i in range(1, 5)]
            list_of_parameters = [json.load(open(os.path.join(path_to_params, f'{modelo_seleccionado}_params_model_{i}.json'))) for i in range(1, 5)]
            
            # Lo ponemos en una tupla.
            tuple_of_dict_dfs_to_predict_and_month = (dict_of_dfs, month_to_predict)

            # data_to_models_prepared es una lista con 4 tuplas, cada tupla tiene 6 elementos de la forma (X_train, X_val, X_test, y_train, y_val, y_test)
            data_to_models_prepared = initialization_eda_fe(make_predictions_selfmade=True, tuple_of_dict_dfs_to_predict_and_month=tuple_of_dict_dfs_to_predict_and_month)

            # Cargamos los parámetros.
            params_1 = list_of_parameters[0]
            params_2 = list_of_parameters[1]
            params_3 = list_of_parameters[2]
            params_4 = list_of_parameters[3]

            if modelo_seleccionado == 'random_forest':
                rf1 = RandomForestRegressor(**params_1)
                rf_2 = RandomForestRegressor(**params_2)
                rf_3 = RandomForestRegressor(**params_3)
                rf_4 = RandomForestRegressor(**params_4)
                loaded_models = [rf1, rf_2, rf_3, rf_4]

            elif modelo_seleccionado == 'xgboost':
                xgb1 = XGBRegressor(**params_1)
                xgb2 = XGBRegressor(**params_2)
                xgb3 = XGBRegressor(**params_3)
                xgb4 = XGBRegressor(**params_4)
                loaded_models = [xgb1, xgb2, xgb3, xgb4]

            elif modelo_seleccionado == 'lightgbm':
                lgb1 = LGBMRegressor(**params_1)
                lgb2 = LGBMRegressor(**params_2)
                lgb3 = LGBMRegressor(**params_3)
                lgb4 = LGBMRegressor(**params_4)
                loaded_models = [lgb1, lgb2, lgb3, lgb4]

            elif modelo_seleccionado == 'gradient_boosting':
                gb1 = GradientBoostingRegressor(**params_1)
                gb2 = GradientBoostingRegressor(**params_2)
                gb3 = GradientBoostingRegressor(**params_3)
                gb4 = GradientBoostingRegressor(**params_4)
                loaded_models = [gb1, gb2, gb3, gb4]
                
             # Inicializa la lista para almacenar los scores SMAPE
            smape_scores = []
            # Entrena y evalúa cada modelo
            for i, model in enumerate(loaded_models):
                model.fit(data_to_models_prepared[i][0], data_to_models_prepared[i][3])
                y_pred = model.predict(data_to_models_prepared[i][2])
                y_true = data_to_models_prepared[i][5]
                smape = Metricas.smape(y_true, y_pred)
                smape_scores.append(smape)

            # Calcula el SMAPE promedio
            smape_avg = sum(smape_scores) / len(smape_scores)
            smape_results = {'model_smapes': smape_scores, 'average_smape': smape_avg}

            return render_template('resultado.html', smape_results=smape_results)
            
    except Exception as e:
        return f"Ocurrió un error: {str(e)}", 500

if __name__ == '__main__':
    app.run(debug=True)