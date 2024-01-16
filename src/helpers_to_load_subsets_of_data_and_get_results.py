from results import *
from utils import *
from eda import *
from fe import *

def carga_datos_si_json_si_params_month_24(ruta_outputs : str = os.path.join(os.getcwd(), '..', 'data', 'outputs', 'modelos_joblibs'),
                                           type_of_model : str = 'random_forest'
                                           ):

    # Cargamos los datos de la carpeta de outputs.
    RUTA_DE_LOS_OUTPUTS = ruta_outputs 
    
    list_of_models = []
    list_of_params = []

    name_of_file_of_joblib_1 = f'{type_of_model}_model_1.joblib'
    name_of_file_of_joblib_2 = f'{type_of_model}_model_2.joblib'
    name_of_file_of_joblib_3 = f'{type_of_model}_model_3.joblib'
    name_of_file_of_joblib_4 = f'{type_of_model}_model_4.joblib'

    name_of_file_of_params_1 = f'{type_of_model}_optuna_params_1.json'
    name_of_file_of_params_2 = f'{type_of_model}_optuna_params_2.json'
    name_of_file_of_params_3 = f'{type_of_model}_optuna_params_3.json'
    name_of_file_of_params_4 = f'{type_of_model}_optuna_params_4.json'

    # Cargamos los modelos.
    modelo_1 = joblib.load(os.path.join(RUTA_DE_LOS_OUTPUTS, name_of_file_of_joblib_1))
    modelo_2 = joblib.load(os.path.join(RUTA_DE_LOS_OUTPUTS, name_of_file_of_joblib_2))
    modelo_3 = joblib.load(os.path.join(RUTA_DE_LOS_OUTPUTS, name_of_file_of_joblib_3))
    modelo_4 = joblib.load(os.path.join(RUTA_DE_LOS_OUTPUTS, name_of_file_of_joblib_4))

    # Cargamos los parametros.
    params_1 = json.load(open(os.path.join(RUTA_DE_LOS_OUTPUTS, name_of_file_of_params_1)))
    params_2 = json.load(open(os.path.join(RUTA_DE_LOS_OUTPUTS, name_of_file_of_params_2)))
    params_3 = json.load(open(os.path.join(RUTA_DE_LOS_OUTPUTS, name_of_file_of_params_3)))
    params_4 = json.load(open(os.path.join(RUTA_DE_LOS_OUTPUTS, name_of_file_of_params_4)))
                
    list_of_models = [modelo_1, modelo_2, modelo_3, modelo_4]
    list_of_params = [params_1, params_2, params_3, params_4]
    
    return list_of_models, list_of_params


def cargar_datos_para_un_mes_y_obtener_la_tupla(mes_a_predecir : int = 0, seed : int = 42) -> Tuple[np.ndarray]:
    
    data_loader = DataLoader()
    dict_of_dfs: Dict[str, pd.DataFrame] = {}
    dict_of_dfs['proteins'], dict_of_dfs['peptides'], dict_of_dfs['clinical'], _ = data_loader.load_train_data()
    tuple_of_dict_dfs_to_predict_and_month = (dict_of_dfs, mes_a_predecir)

    # Obtengo la tupla de entrada de los modelos para entrenarlos o realizar prediccion con los mejores parametros.
    data_to_models_prepared = initialization_eda_fe(make_predictions_selfmade=True,
                                                    tuple_of_dict_dfs_to_predict_and_month=tuple_of_dict_dfs_to_predict_and_month,
                                                    seed=seed)

    return data_to_models_prepared


def entrena_modelos_y_genera_graficas(modelos, 
                                      params, 
                                      data_sets,
                                      title = "Learning Curves",
                                      verbose = False,
                                      ruta_outputs = PathStorage().DIR_IMAGES
                                      ):
    
    cv = KFold(n_splits=5, shuffle=True, random_state=42)
    
    for i, (model, param, data_set) in enumerate(zip(modelos, params, data_sets), 1):
        X_train, _, X_test, y_train, _, y_test = data_set
        model.set_params(**param)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        
        plot_learning_curve(model, title, X_train, y_train, cv=cv, n_jobs=4, ruta_outputs=ruta_outputs, updrs_number=i, type_of_model="random_forest", verbose=verbose)
        plot_residuals_vs_fitted(y_test, pred, ruta_outputs=ruta_outputs, updrs_number=i, type_of_model="random_forest", verbose=verbose)

def entrena_y_predice(modelos, params, data_sets):
    predictions = []
    smape_scores = []
    rmse_scores = []
    
    for model, param, data_set in zip(modelos, params, data_sets):
        X_train, X_val, X_test, y_train, y_val, y_test = data_set
        model.set_params(**param)
        model.fit(X_train, y_train)
        pred = model.predict(X_test)
        predictions.append(pred)
        smape_scores.append(Metricas.smape(y_test, pred))
        rmse_scores.append(Metricas.rmse(y_test, pred))
    
    return predictions, smape_scores, rmse_scores


def cargar_datos_mas_eda_fe(mes_a_predecir = 24):

    data_loader = DataLoader() 
    dict_of_dfs : Dict[str, pd.DataFrame] = {}

    dict_of_dfs['proteins'], dict_of_dfs['peptides'], dict_of_dfs['clinical'], _ = data_loader.load_train_data()

    month_to_filter = mes_a_predecir

    df_proteins = dict_of_dfs['proteins']
    df_peptides = dict_of_dfs['peptides']
    df_clinical = dict_of_dfs['clinical']

    # Dataframe filtered by month.
    proteins_filtered_by_month = EdaNew.filter_proteins_by_month(df_proteins, month_to_filter=month_to_filter)
    peptides_filtered_by_month = EdaNew.filter_peptides_by_month(df_peptides, month_to_filter=month_to_filter)
    clinical_filtered_by_month = EdaNew.filter_clinical_by_month(df_clinical, month_to_filter=month_to_filter)

    # Merging the normal three entry dataframes.
    proteins_peptides_merged_month_24 = pd.merge(peptides_filtered_by_month, proteins_filtered_by_month, how='left', on=['visit_id', 'visit_month', 'patient_id', 'UniProt'])
    df_merged_all = pd.merge(proteins_peptides_merged_month_24, clinical_filtered_by_month, how='left', on=['visit_id', 'patient_id', 'visit_month'])

    # Removing duplicates (Also you can analyze or visualize).
    df_merged_all_filtered_by_duplicates = EdaNew.analyze_and_visualize_duplicates(df_merged_all, verbose=False)

    # Remove null values.
    df_merged_filtered_by_nullsand_duplicates = EdaNew.calculate_and_remove_null_values(df_merged_all_filtered_by_duplicates, verbose=False)

    #
    df_merged_all_filtered_without_ouliers_iqr = EdaNew.remove_outliers_iqr(df_merged_filtered_by_nullsand_duplicates)

    #
    df_merged_all_filtered_with_log_column = EdaNew.add_log_columns(df_merged_all_filtered_without_ouliers_iqr)

    # 
    df_merged_all_filtered_without_ouliers_iqr = EdaNew.remove_outliers_iqr(df_merged_all_filtered_with_log_column)

    #
    df_merged_all_filtered_full_Eda = EdaNew.remove_outliers_std(df_merged_all_filtered_without_ouliers_iqr)

    # 
    df_merged_all_filtered_full_Eda = EdaNew.drop_null_count(df_merged_all_filtered_full_Eda)

    #
    df_merged_all_filtered_full_Eda = EdaNew.drop_upd23b_clinical_state_on_medication(df_merged_all_filtered_full_Eda)

    #
    df_merged_all_filtered_full_Eda = EdaNew.drop_visit_id_and_visit_month(df_merged_all_filtered_full_Eda)

    df_after_eda = df_merged_all_filtered_full_Eda

    df_after_fe = full_FeatureEngineeringNew(df_after_eda) # TODO

    # data_to_models = df_after_fe.drop(columns=['patient_id', 'UniProt', 'Peptide'], axis=1) # TODO

    data_to_models = df_after_fe

    train_results = [data_to_models.drop(columns=[f'updrs_{j}' for j in range(1, 5) if j != i]) for i in range(1, 5)]

    return train_results