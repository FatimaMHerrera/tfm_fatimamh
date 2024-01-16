import pandas as pd
import numpy as np
from typing import List, Dict


class FeatureEngineering:
    def __init__(self, proteins, peptides, clinical):
        self.proteins = proteins
        self.peptides = peptides
        self.clinical = clinical

    def calculate_statistics(self, df, value_column):
        stats_df = df.groupby('patient_id')[value_column].agg(['max', 'min', 'mean', 'std']).reset_index()
        stats_df.columns = ['patient_id', f'{value_column}_max', f'{value_column}_min', f'{value_column}_mean', f'{value_column}_std']
        return stats_df

    def merge_with_stats(self, original_df, stats_df):
        return pd.merge(original_df, stats_df, on='patient_id', how='left')

    def merge_all_data(self):
        proteins_stats = self.calculate_statistics(self.proteins, 'NPX')
        peptides_stats = self.calculate_statistics(self.peptides, 'PeptideAbundance')

        proteins_with_stats = self.merge_with_stats(self.proteins, proteins_stats)
        peptides_with_stats = self.merge_with_stats(self.peptides, peptides_stats)

        full_merged_df = pd.merge(pd.merge(proteins_with_stats, peptides_with_stats, on='patient_id', how='outer'), self.clinical, on='patient_id', how='outer')
        return full_merged_df

    def clean_data(self, df):
        return df.dropna()

class FeatureEngineeringNew:

    __slots__ = ['df']

    def __init__(self, df : pd.DataFrame) -> None:
        self.df = df
    
    def create_peptide_per_protein(self):
        self.df['Peptide_per_Protein'] = self.df['PeptideAbundance'] / self.df['NPX']
        return self.df
    
    # TODO to_log : bool = True (revisado)
    def columns_related_to_npx(self, to_log : bool = True, properties_on : str = 'all', percentil_option : bool = True, stats : List[str] = ['mean', 'median', 'min', 'max', 'std', 'var']):
        
        if to_log:
            column = 'NPX_log'
        else:
            column = 'NPX'

        self.df[f'{column}_percentile_in_{properties_on}'] = self.df[column].rank(pct=percentil_option)

        npx_stats = self.df[column].agg(stats)
        for stat in stats:
            self.df[f'{column}_{stat}_in_{properties_on}'] = npx_stats[stat]
            
        return self.df
    
    # TODO to_log : bool = True (revisado)
    def columns_related_to_peptide_abundance(self, to_log : bool = True, properties_on : str = 'local', percentil_option : bool = True, stats : List[str] = ['mean', 'median', 'min', 'max', 'std', 'var']):

        if to_log:
            column = 'PeptideAbundance_log'
        else:
            column = 'PeptideAbundance'

        self.df[f'{column}_percentile_in_{properties_on}'] = self.df.groupby('UniProt')[column].rank(pct=percentil_option)

        for stat in stats:
            self.df[f'{column}_{stat}_in_UniProt'] = self.df.groupby('UniProt')[column].transform(stat)
        
        return self.df
    
    def columns_related_to_updrs(self, percentil_option: bool = True, stats: List[str] = ['mean', 'median', 'min', 'max', 'std', 'var']):
        for column_name in self.df.columns:
            if column_name.startswith('updrs'):
                for stat in stats:
                    new_column_name = f'{column_name}_{stat}'  # Nombre de la nueva columna
                    if stat == 'percentile' and percentil_option:
                        percentile_values = [10, 25, 50, 75, 90]  # Valores de percentil
                        percentiles = self.df[column_name].quantile([0.1, 0.25, 0.5, 0.75, 0.9])
                        for i, percentile in enumerate(percentile_values):
                            self.df[new_column_name + f'_p{percentile}'] = percentiles[i]
                    else:
                        self.df[new_column_name] = self.df[column_name].agg(stat)
        return self.df
    
    def columns_related_to_updrs_to_total_stats(self, stats: List[str] = ['mean', 'median', 'min', 'max', 'std', 'var']):
        updrs_cols = [col for col in self.df.columns if col.startswith('updrs')]
        for stat in stats:
            self.df[f'updrs_total_{stat}'] = self.df[updrs_cols].agg(stat, axis=1)
        return self.df

    def remove_categorical_or_objects(self):
        df_copy = self.df.copy()
        df_copy.drop(['patient_id', 'PeptideAbundance', 'NPX', 'UniProt', 'Peptide'], axis=1, inplace=True) #, 'UniProt', 'Peptide'
        return df_copy
    
    def remove_peptide_abundance_npx(self):
        df_copy = self.df.copy()
        df_copy.drop(['PeptideAbundance', 'NPX'], axis=1, inplace=True)
        return df_copy
    
    def remove_patient_id_unitprot_peptide(self):
        df_copy = self.df.copy()
        df_copy.drop(['UniProt', 'Peptide'], axis=1, inplace=True)
        return df_copy
    
    

def full_FeatureEngineeringNew(df : pd.DataFrame):

    fe = FeatureEngineeringNew(df)
    df = fe.create_peptide_per_protein()
    df = fe.columns_related_to_npx()
    df = fe.columns_related_to_peptide_abundance()
    df = fe.columns_related_to_updrs()
    if df.shape[0] == df.patient_id.nunique():
        print(True)
    else:
        print(False)
    # df = fe.columns_related_to_updrs_to_total_stats()
    df = fe.remove_categorical_or_objects()
    # df = fe.remove_peptide_abundance_npx()
    # df = fe.remove_patient_id_unitprot_peptide()


    return df 

        

    
        
