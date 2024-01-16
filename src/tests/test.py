import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
import unittest
import pandas as pd
import random as rnd
import numpy as np
import json
import joblib
from utils import DataLoader
from eda import EdaNew, full_EdaNew
from fe import FeatureEngineeringNew, full_FeatureEngineeringNew
from edafecomp import initialization_eda_fe
from utils import DataFrameOptimizer
from modelling import DataPreparationToModelNew, Metricas, ModelTrainEvaluate, train_and_save_models
from optimization import run_optuna_optimization, early_stopping_callback

def sample_dataframes(dataframes, seed=42, sample_size=5):
    """
    Función para muestrear de forma coherente de múltiples DataFrames basados en una columna común.

    Parámetros:
    - dataframes (dict): Un diccionario de DataFrames.
    - seed (int): Semilla para la generación de números aleatorios.
    - sample_size (int): Número de muestras a seleccionar.

    Retorna:
    - dict: Un diccionario de DataFrames filtrados.
    """
    # Establecer la semilla aleatoria para reproducibilidad
    rnd.seed(seed)

    # Determinar los patient_id únicos
    unique_patient_ids = set()
    for df in dataframes.values():
        unique_patient_ids = unique_patient_ids.union(set(df['patient_id']))

    # Convertir a lista y muestrear
    unique_patient_ids = list(unique_patient_ids)
    sampled_patient_ids = rnd.sample(unique_patient_ids, sample_size)

    # Filtrar cada DataFrame y retornar un nuevo diccionario
    filtered_dfs = {}
    for key, df in dataframes.items():
        filtered_dfs[key] = df[df['patient_id'].isin(sampled_patient_ids)]

    return filtered_dfs

class TestEda(unittest.TestCase):

    def setUp(self):
        
        self.if_sample = False
        self.sample_size = 5
        self.month_to_filter = 24
        
        dict_of_dfs = {}
        dict_of_dfs['proteins'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_proteins.csv'))
        dict_of_dfs['peptides'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_peptides.csv'))
        dict_of_dfs['clinical'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_clinical_data.csv'))

        if self.if_sample:

            sampled_dfs = sample_dataframes(dict_of_dfs, self.sample_size)
            self.proteins = sampled_dfs['proteins']
            self.peptides = sampled_dfs['peptides']
            self.clinical = sampled_dfs['clinical']

        else:
            self.proteins = dict_of_dfs['proteins']
            self.peptides = dict_of_dfs['peptides']
            self.clinical = dict_of_dfs['clinical']

        # Merging the normal three entry dataframes.
        self.proteins_peptides_merged_month_24 = pd.merge(self.peptides, self.proteins, how='left', on=['visit_id', 'visit_month', 'patient_id', 'UniProt'])
        self.df_merged_all = pd.merge(self.proteins_peptides_merged_month_24, self.clinical, how='left', on=['visit_id', 'patient_id', 'visit_month'])

    def test_filter_clinical_by_month(self):

        result_df = EdaNew.filter_clinical_by_month(self.clinical, self.month_to_filter)

        if self.if_sample:
            self.assertEqual(len(result_df), self.sample_size)

        self.assertTrue(all(col in result_df.columns for col in ['patient_id', 'updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']))

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def test_filter_proteins_by_month(self):

        result_df = EdaNew.filter_proteins_by_month(self.proteins, self.month_to_filter)

        self.assertTrue(all(col in result_df.columns for col in ['visit_id', 'visit_month', 'patient_id', 'UniProt', 'NPX']))

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def test_filter_peptides_by_month(self):

        result_df = EdaNew.filter_peptides_by_month(self.peptides, self.month_to_filter)

        self.assertTrue(all(col in result_df.columns for col in ['visit_id', 'visit_month', 'patient_id', 'UniProt', 'Peptide', 'PeptideAbundance']))

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def test_analyze_and_visualize_duplicates(self):

        result_df = EdaNew.analyze_and_visualize_duplicates(self.df_merged_all, verbose=False)

        self.assertTrue(result_df.duplicated(keep=False).sum()==0)

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def test_calculate_and_remove_null_values(self):

        result_df = EdaNew.calculate_and_remove_null_values(self.df_merged_all, verbose=False)
        self.df = result_df
        self.assertTrue(result_df.isnull().sum().sum()==0)

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def remove_outliers_iqr(self):

        result_df = EdaNew.remove_outliers_iqr(self.df_merged_all)

        self.assertTrue(result_df.isnull().sum().sum()==0)
        self.assertTrue(len(result_df) < len(self.df_merged_all))

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def test_add_log_columns(self):

        result_df = EdaNew.add_log_columns(self.df_merged_all)

        self.assertTrue(all(col in result_df.columns for col in ['NPX_log', 'PeptideAbundance_log']))

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def test_remove_outliers_std(self):
        original_row_count = len(self.df_merged_all)

        # Calcular la cantidad inicial de NaN
        initial_nans = self.df_merged_all.isnull().sum().sum()

        # Aplicar la función
        result_df = EdaNew.remove_outliers_std(self.df_merged_all)

        # Calcular la proporción de filas eliminadas
        rows_removed = original_row_count - len(result_df)
        proportion_removed = rows_removed / original_row_count

        # Calcular la cantidad de NaN después de aplicar la función
        final_nans = result_df.isnull().sum().sum()

        # Calcular la diferencia en la cantidad de NaN
        additional_nans = final_nans - initial_nans

        # Calcular la proporción total de NaN adicionales generados
        total_values = np.prod(result_df.shape)
        proportion_additional_nans = additional_nans / total_values

        threshold_rows = 0.1  # Umbral del 10% para filas eliminadas
        threshold_nans = 0.2  # Umbral del 10% para NaN adicionales generados # TODO (Hay Nans)

        # Asegurar que no más del 10% de las filas fueron eliminadas
        self.assertLessEqual(proportion_removed, threshold_rows, f"More than {threshold_rows * 100}% of the rows were removed.")

        # Asegurar que no más del 10% de los valores adicionales son NaN
        self.assertLessEqual(proportion_additional_nans, threshold_nans, f"More than {threshold_nans * 100}% of additional data is NaN.")

        test_name = self.id()
        print(f'\nTest passed: {test_name}')


    def test_drop_upd23b_clinical_state_on_medication(self):

        result_df = EdaNew.drop_upd23b_clinical_state_on_medication(self.df_merged_all)

        self.assertFalse(all(col in result_df.columns for col in ['upd23b_clinical_state_on_medication']))

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def test_drop_visit_id_and_visit_month(self):

        result_df = EdaNew.drop_visit_id_and_visit_month(self.df_merged_all)

        self.assertFalse(all(col in result_df.columns for col in ['visit_id', 'visit_month']))

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

    def test_full_EdaNew(self):

        # Preparar los datos de entrada
        dict_of_dfs = {
            'proteins': self.proteins,
            'peptides': self.peptides,
            'clinical': self.clinical
        }

        # Aplicar la función full_EdaNew
        result_df = full_EdaNew(dict_of_dfs, self.month_to_filter)

        # Verificar que las columnas esperadas estén presentes después de filtrar por mes
        expected_columns_after_filter = ['patient_id', 'updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'] # Completa con las columnas esperadas
        self.assertTrue(all(col in result_df.columns for col in expected_columns_after_filter))

        # Verificar que no hay duplicados
        self.assertTrue(result_df.duplicated().sum() == 0)

        # Verificar que no hay valores nulos
        self.assertTrue(result_df.isnull().sum().sum() == 0)

        # Verificar la presencia de columnas logarítmicas
        self.assertTrue('NPX_log' in result_df.columns and 'PeptideAbundance_log' in result_df.columns)

        # Verificar la eliminación de ciertas columnas
        self.assertFalse('upd23b_clinical_state_on_medication' in result_df.columns)
        self.assertFalse('visit_id' in result_df.columns and 'visit_month' in result_df.columns)

        test_name = self.id()
        print(f'\nTest passed: {test_name}')

class TestFe(unittest.TestCase):


    def setUp(self):

        self.if_sample = False
        self.sample_size = 5
        self.month_to_filter = 24
        
        dict_of_dfs = {}
        dict_of_dfs['proteins'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_proteins.csv'))
        dict_of_dfs['peptides'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_peptides.csv'))
        dict_of_dfs['clinical'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_clinical_data.csv'))

        self.df_after_eda = full_EdaNew(dict_of_dfs)
        

    def test_create_peptide_per_protein(self):
        fe = FeatureEngineeringNew(self.df_after_eda)
        result_df = fe.create_peptide_per_protein()
        self.assertIn('Peptide_per_Protein', result_df.columns)

        print(f'\nTest passed: {self.id()}')

    def test_columns_related_to_npx(self):
        fe = FeatureEngineeringNew(self.df_after_eda)
        result_df = fe.columns_related_to_npx()
        expected_columns = ['NPX_log_percentile_in_all', 'NPX_log_mean_in_all']
        for col in expected_columns:
            self.assertIn(col, result_df.columns)

        print(f'\nTest passed: {self.id()}')

    def test_full_FeatureEngineeringNew(self):
        result_df = full_FeatureEngineeringNew(self.df_after_eda)

        expected_columns = ['NPX_log_percentile_in_all', 'NPX_log_mean_in_all']
        for col in expected_columns:
            self.assertIn(col, result_df.columns)
        
        # self.assertTrue(result_df.duplicated(keep=False).sum()==0) # TODO (in review)
        self.assertTrue(result_df.shape[0] == self.df_after_eda.shape[0])
        self.assertTrue(result_df.isnull().sum().sum()==0)

        print(f'\nTest passed: {self.id()}')

class TestPreModel(unittest.TestCase):

    def setUp(self):

        self.if_sample = False
        self.sample_size = 5
        self.month_to_filter = 24
        
        dict_of_dfs = {}
        dict_of_dfs['proteins'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_proteins.csv'))
        dict_of_dfs['peptides'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_peptides.csv'))
        dict_of_dfs['clinical'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_clinical_data.csv'))

        self.df_after_eda = full_EdaNew(dict_of_dfs)

        self.df_after_fe = full_FeatureEngineeringNew(self.df_after_eda)

        self.train_results = [DataFrameOptimizer(self.df_after_fe.drop(columns=[f'updrs_{j}' for j in range(1, 5) if j != i]))() for i in range(1, 5)]

        self.data_to_models_preparation = DataPreparationToModelNew(self.train_results, [f'updrs_{i+1}' for i in range(len(self.train_results))])()

    def test_Optimizador_dataframe(self):

        self.assertTrue(self.train_results[0].shape[0] == self.train_results[1].shape[0] == self.train_results[2].shape[0] == self.train_results[3].shape[0])
        self.assertTrue(self.train_results[0].shape[1] == self.train_results[1].shape[1] == self.train_results[2].shape[1] == self.train_results[3].shape[1])
        self.assertTrue(self.train_results[0].isnull().sum().sum() == self.train_results[1].isnull().sum().sum() == self.train_results[2].isnull().sum().sum() == self.train_results[3].isnull().sum().sum() == 0)
        self.assertTrue(self.train_results[0].isna().sum().sum() == 0)
        # self.assertTrue(self.df_after_fe.duplicated(keep=False).sum()==0) # TODO (in review)

class TestModellingNoParalell(unittest.TestCase):

    def setUp(self):

        self.if_sample = False
        self.sample_size = 5
        self.month_to_filter = 24
        self.type_of_model = 'random_forest'
        self.option_parallel = True
        self.seed = 42
        
        self.dict_of_dfs = {}
        self.dict_of_dfs['proteins'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_proteins.csv'))
        self.dict_of_dfs['peptides'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_peptides.csv'))
        self.dict_of_dfs['clinical'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_clinical_data.csv'))

        self.tuple_of_dict_dfs_to_predict_and_month = (self.dict_of_dfs, self.month_to_filter)

        self.make_predictions_selfmade=False
        self.tuple_of_dict_dfs_to_predict_and_month=self.tuple_of_dict_dfs_to_predict_and_month
        self.if_sample=self.if_sample
        self.sample_size=self.sample_size
        self.seed=self.seed
        self.include_all_updrs_in_train=False
        self.save_results = False
        # self.output_path_of_models = os.path.join(os.getcwd(), '..', '..', 'data', 'outputs', 'modelos_joblibs')

        self.df_after_eda = full_EdaNew(self.tuple_of_dict_dfs_to_predict_and_month[0])

        self.df_after_fe = full_FeatureEngineeringNew(self.df_after_eda)

        self.train_results = [DataFrameOptimizer(self.df_after_fe.drop(columns=[f'updrs_{j}' for j in range(1, 5) if j != i]))() for i in range(1, 5)]

        self.data_to_models_preparation = DataPreparationToModelNew(self.train_results, [f'updrs_{i+1}' for i in range(len(self.train_results))])()

        # initialization_eda_fe

        PARAMETERS_TO_MODELS = {
            self.type_of_model : { # case 'random_forest'
                "n_estimators": 156, 
                "max_depth": 28, 
                "min_samples_split": 20, 
                "min_samples_leaf": 20, 
                "max_features": "log2", 
                "max_leaf_nodes": 985
            }
        }
        
        # Ruta del archivo de parámetros
        params_path = os.path.join(DataLoader.DIR_MODELS, f'{self.type_of_model}_optuna_params.json')

        # Verificar si existe el archivo de parámetros
        if os.path.exists(params_path):
            with open(params_path, 'r') as json_file:
                model_params = json.load(json_file)
        else:
            # Utilizar parámetros predeterminados si no existen parámetros optimizados
            model_params = PARAMETERS_TO_MODELS.get(self.type_of_model, {})

        # Instanciar ModelTrainEvaluate con los parámetros cargados
        model_train_evaluate = ModelTrainEvaluate(type_of_model=self.type_of_model, 
                                                  model_params=model_params, 
                                                  option_parallel=self.option_parallel, 
                                                  seed=self.seed
                                                  )

        # Entrenar el modelo y obtener resultados
        model_train_evaluate.obtaining_the_results(self.data_to_models_preparation)

        self.output_path_of_models = os.path.join(os.getcwd(), '..', '..', 'data', 'outputs', 'modelos_joblibs')

        # Obtener y guardar el mejor modelo
        best_model = model_train_evaluate.save_best_models(self.output_path_of_models)

        if self.save_results:
            model_file_path = self.output_path_of_models
            joblib.dump(best_model, model_file_path)
        else:
            self.model = best_model

    def test_ModelTrainingNew(self):
        pass

    def test_Metricas(self):
        pass

class TestModellingParalell(unittest.TestCase):
    pass

class TestModellingOptunaParalell(unittest.TestCase):

    def setUp(self):
        # Configuración inicial similar a otras clases de prueba
        # Cargar datos, preparar entorno, etc.
        self.if_sample = True
        self.sample_size = 5  # Puedes ajustar este tamaño según tus necesidades
        self.type_of_model = 'random_forest'
        self.option_parallel = True
        self.seed = 42
        self.month_to_filter = 24

        # Asumiendo que ya tienes una función que prepara los datos para el modelo
        self.data_to_models_prepared = self.prepare_model_data()

    def prepare_model_data(self):
        
        self.dict_of_dfs = {}
        self.dict_of_dfs['proteins'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_proteins.csv'))
        self.dict_of_dfs['peptides'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_peptides.csv'))
        self.dict_of_dfs['clinical'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_clinical_data.csv'))

        self.tuple_of_dict_dfs_to_predict_and_month = (self.dict_of_dfs, self.month_to_filter)

        self.make_predictions_selfmade=False
        self.tuple_of_dict_dfs_to_predict_and_month=self.tuple_of_dict_dfs_to_predict_and_month
        self.if_sample=self.if_sample
        self.sample_size=self.sample_size
        self.seed=self.seed
        self.include_all_updrs_in_train=False
        self.save_results = False
        self.output_path_of_models = os.path.join(os.getcwd(), '..', '..', 'data', 'outputs', 'modelos_joblibs')

        self.df_after_eda = full_EdaNew(self.tuple_of_dict_dfs_to_predict_and_month[0])

        self.df_after_fe = full_FeatureEngineeringNew(self.df_after_eda)

        self.train_results = [DataFrameOptimizer(self.df_after_fe.drop(columns=[f'updrs_{j}' for j in range(1, 5) if j != i]))() for i in range(1, 5)]

        self.data_to_models_preparation = DataPreparationToModelNew(self.train_results, [f'updrs_{i+1}' for i in range(len(self.train_results))])()
        
        return self.data_to_models_preparation

    def test_optuna_optimization(self):
        # Llama a la función de optimización de Optuna y verifica los resultados
        try:
            run_optuna_optimization(self.data_to_models_prepared, 
                                    self.type_of_model, 
                                    self.option_parallel, 
                                    self.seed,
                                    self.output_path_of_models)
            self.assertTrue(True)
        except Exception as e:
            self.fail(f"Optuna optimization failed: {e}")

class TestModellingOptunaNoParalell(unittest.TestCase):
    pass

# class TestOptunaParallelWithSeeds(unittest.TestCase):

#     def setUp(self):
#         # Configuración inicial similar a otras clases de prueba
#         # Cargar datos, preparar entorno, etc.
#         self.if_sample = True
#         self.sample_size = 5  # Puedes ajustar este tamaño según tus necesidades
#         self.type_of_model = 'random_forest'
#         self.option_parallel = True
#         self.seed = 42
#         self.month_to_filter = 24
#         self.seeds = [42, 123, 789]

#         # Asumiendo que ya tienes una función que prepara los datos para el modelo
#         self.data_to_models_prepared = self.prepare_model_data()

#     def prepare_model_data(self):
        
#         self.dict_of_dfs = {}
#         self.dict_of_dfs['proteins'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_proteins.csv'))
#         self.dict_of_dfs['peptides'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_peptides.csv'))
#         self.dict_of_dfs['clinical'] = pd.read_csv(os.path.join(os.getcwd(), '..', '..', 'data', 'inputs', 'train_clinical_data.csv'))

#         self.tuple_of_dict_dfs_to_predict_and_month = (self.dict_of_dfs, self.month_to_filter)

#         self.make_predictions_selfmade=False
#         self.tuple_of_dict_dfs_to_predict_and_month=self.tuple_of_dict_dfs_to_predict_and_month
#         self.if_sample=self.if_sample
#         self.sample_size=self.sample_size
#         self.seed=self.seed
#         self.include_all_updrs_in_train=False
#         self.save_results = False
#         self.output_path_of_models = os.path.join(os.getcwd(), '..', '..', 'data', 'outputs', 'modelos_joblibs')

#         self.df_after_eda = full_EdaNew(self.tuple_of_dict_dfs_to_predict_and_month[0])

#         self.df_after_fe = full_FeatureEngineeringNew(self.df_after_eda)

#         self.train_results = [DataFrameOptimizer(self.df_after_fe.drop(columns=[f'updrs_{j}' for j in range(1, 5) if j != i]))() for i in range(1, 5)]

#         self.data_to_models_preparation = DataPreparationToModelNew(self.train_results, [f'updrs_{i+1}' for i in range(len(self.train_results))])()
        
#         return self.data_to_models_preparation  # Agrega las semillas que desees probar

#     def run_optimization_with_seed(self, seed):
#         # Ejecuta la optimización de Optuna con una semilla específica y retorna el resultado
#         try:
#             result = run_optuna_optimization(self.data_to_models_prepared, 
#                                             self.type_of_model, 
#                                             self.option_parallel, 
#                                             seed,
#                                             self.output_path_of_models
#                                             )
#             return result
#         except Exception as e:
#             self.fail(f"Optuna optimization with seed {seed} failed: {e}")

#     def test_optuna_optimization_with_seeds(self):
#         # Ejecuta la optimización para cada semilla y guarda los resultados
#         results = []
#         for seed in self.seeds:
#             result = self.run_optimization_with_seed(seed)
#             results.append(result)

#         # Calcula el promedio de los resultados o realiza otras estadísticas
#         average_result = sum(results) / len(results)
        
#         # Puedes realizar aserciones para verificar el promedio o cualquier otra estadística que desees
#         self.assertTrue(average_result > 0.5)  # Reemplaza esto con tus propias aserciones
        
#         # Imprime los resultados si es necesario
#         print(f"Average result for seeds {self.seeds}: {average_result}")


if __name__ == '__main__':
    unittest.main()


