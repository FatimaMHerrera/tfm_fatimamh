import random as rnd
import numpy as np
import pandas as pd

from typing import Dict, List, Any, Tuple

from utils import (DataLoader, DataFrameOptimizer, reduce_mem_usage, timing_decorator)
from eda import (EdaNew, full_EdaNew)
from fe import (FeatureEngineeringNew, full_FeatureEngineeringNew)
from modelling import DataPreparationToModelNew


def initialization_eda_fe(make_predictions_selfmade : bool = False, 
                          tuple_of_dict_dfs_to_predict_and_month : Tuple[Dict[str, pd.DataFrame], int] = (), 
                          if_sample: bool = False, 
                          sample_size : int = 50, 
                          seed : int = 42, 
                          include_all_updrs_in_train : bool = False, 
                          month_to_filter : int = 42
                          ) -> List[Tuple[pd.DataFrame]]:

    rnd.seed(seed)
    np.random.seed(seed)

    if make_predictions_selfmade == False:
        dict_of_dfs : Dict[str, pd.DataFrame] = {}

        dict_of_dfs['proteins'], dict_of_dfs['peptides'], dict_of_dfs['clinical'], _ = DataLoader().load_train_data()
    
    elif make_predictions_selfmade == True:
        dict_of_dfs = tuple_of_dict_dfs_to_predict_and_month[0]
        month_to_filter = tuple_of_dict_dfs_to_predict_and_month[1]

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

    data_to_models_preparation = DataPreparationToModelNew(train_results, [f'updrs_{i+1}' for i in range(len(train_results))], random_state=seed)()

    return data_to_models_preparation

