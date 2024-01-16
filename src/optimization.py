import os
import json
import joblib
import optuna
from utils import DataLoader, PathStorage
from modelling import ModelTrainEvaluate, Metricas

def early_stopping_callback(stopping_rounds, tolerance=1e-5):
    best_value = None
    counter = 0

    def callback(study, trial):
        nonlocal best_value, counter
        if best_value is None or study.best_value < best_value - tolerance:
            best_value = study.best_value
            counter = 0
        else:
            counter += 1
            if counter >= stopping_rounds:
                study.stop()

    return callback

def run_optuna_optimization(data_to_models_prepared, type_of_model, option_parallel, seed, path_to_save_models=PathStorage().DIR_MODELS_JOBLIBS):
    
    def objective(trial, data_set):
        if type_of_model == 'xgboost':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 1000),  # Rango más amplio
                'max_depth': trial.suggest_int('max_depth', 1, 50),  # Permitir árboles más simples y más complejos
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 50),  # Mayor variabilidad en el tamaño mínimo de las divisiones
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 50),  # Variar más el tamaño mínimo de las hojas
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2', None]),  # Incluir la opción de usar todas las características
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 2000, log=True)  # Rango más amplio con escala logarítmica
            }

        elif type_of_model == 'random_forest':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 50, 500),
                'max_depth': trial.suggest_int('max_depth', 5, 50),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 20),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 20),
                'max_features': trial.suggest_categorical('max_features', ['sqrt', 'log2']),
                'max_leaf_nodes': trial.suggest_int('max_leaf_nodes', 10, 1000, log=True)
            }

        elif type_of_model == 'gradient_boosting':
            model_params = {
                'n_estimators': trial.suggest_int('n_estimators', 10, 100),
                'max_depth': trial.suggest_int('max_depth', 2, 32, log=True),
                'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.1, log=True),
                'subsample': trial.suggest_float('subsample', 0.5, 1.0),
                'min_samples_split': trial.suggest_int('min_samples_split', 2, 14),
                'min_samples_leaf': trial.suggest_int('min_samples_leaf', 1, 14)
            }

        elif type_of_model == "lightgbm":
            model_params = {
                "n_estimators": trial.suggest_int("n_estimators", 100, 1000),
                "max_depth": trial.suggest_int("max_depth", 3, 15),
                "learning_rate": trial.suggest_float("learning_rate", 0.01, 0.1, log=True),
                "subsample": trial.suggest_float("subsample", 0.5, 1.0),
                "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
                "reg_alpha": trial.suggest_float("reg_alpha", 0, 5),
                "reg_lambda": trial.suggest_float("reg_lambda", 0, 5),
                "num_leaves": trial.suggest_int("num_leaves", 2, 256),
                "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            }

        model_train_evaluate = ModelTrainEvaluate(
            type_of_model=type_of_model, 
            model_params=model_params, 
            option_parallel=option_parallel, 
            seed=seed
        )

        model_train_evaluate.obtaining_the_results([data_set])  # Solo un conjunto de datos

        smape_values = model_train_evaluate.results[type_of_model]['smape']
        mean_smape = Metricas.calculate_mean_of_metrics(smape_values)
        return mean_smape

    # Nueva función para optimizar un conjunto de datos individual
    def optimize_for_data_set(data_set):
        study = optuna.create_study(direction='minimize')
        study.optimize(lambda trial: objective(trial, data_set), n_trials=20, callbacks=[early_stopping_callback(5)])
        return study.best_params, study.best_value

    for i, data_set in enumerate(data_to_models_prepared):
        best_params, best_value = optimize_for_data_set(data_set)

        # Guardar los mejores parámetros
        params_file_path = os.path.join(path_to_save_models, f'{type_of_model}_optuna_params_{i+1}.json')
        with open(params_file_path, 'w') as json_file:
            json.dump(best_params, json_file)

        # Instanciar y entrenar el mejor modelo, incluyendo el índice del modelo
        best_model_train_evaluate = ModelTrainEvaluate(
            type_of_model=type_of_model, 
            model_params=best_params, 
            option_parallel=option_parallel, 
            seed=seed
        )
        
        best_model_train_evaluate.train_and_save_individual_models([data_set], path_to_save_models, i + 1)

        print(f"Mejor modelo para UPDRS {i+1} guardado en {path_to_save_models} con SMAPE: {best_value}")
