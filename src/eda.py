import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt

from typing import List, Dict, Tuple


class EdaNew:

    @staticmethod
    def filter_clinical_by_month(clinical_df : pd.DataFrame, month_to_filter : int = 24, updrs_cols : List[str] = ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4']) -> pd.DataFrame:
        """
        Filter the clinical data by month and select the columns of interest.
        """

        clinical_df_filtered_by_month = clinical_df[clinical_df.visit_month == month_to_filter]

        return clinical_df_filtered_by_month
    
    @staticmethod
    def filter_proteins_by_month(proteins_df: pd.DataFrame, month_to_filter : int = 24) -> pd.DataFrame:
        """
        Filter the proteins data by month and select the columns of interest.
        """

        proteins_df_filtered_by_month = proteins_df[proteins_df.visit_month == month_to_filter]

        return proteins_df_filtered_by_month
    
    def filter_peptides_by_month(peptides_df: pd.DataFrame, month_to_filter : int = 24) -> pd.DataFrame:
        """
        Filter the peptides data by month and select the columns of interest.
        """

        peptides_df_filtered_by_month = peptides_df[peptides_df.visit_month == month_to_filter]

        return peptides_df_filtered_by_month

    @staticmethod
    def analyze_and_visualize_duplicates(data, title :str = 'DataFrame', verbose : bool = False): # TODO analizar y visualizar duplicados.
        """
        Función para analizar y opcionalmente visualizar y eliminar datos duplicados en un DataFrame basándose en columnas específicas.

        Args:
        data (pd.DataFrame): DataFrame a analizar.
        title (str): Título para la visualización.
        index_cols (list): Lista de columnas para identificar duplicados.
        verbose (bool): Si es True, imprime información y muestra gráficos.

        Returns:
        pd.DataFrame: DataFrame con duplicados eliminados.
        """

        duplicate_rows = data.duplicated(keep=False)
        num_duplicate_rows = duplicate_rows.sum()
        proportion_duplicates = num_duplicate_rows / len(data) * 100

        # Eliminar duplicados
        data_filtered = data.drop_duplicates()

        if verbose:
            # Gráfico
            plt.figure(figsize=(10, 4))
            sns.countplot(x=duplicate_rows)
            plt.title(f'Duplicate Counts in {title}')
            plt.ylabel('Count')
            plt.xlabel('Is Duplicate')

            # Mostrar información de duplicados
            print(f'{title} - Proportion of Duplicates: {proportion_duplicates:.2f}%')
            print(f'{title} - Number of Duplicate Rows: {num_duplicate_rows}')
            num_rows_removed = len(data) - len(data_filtered)
            print(f'{title} - Number of Rows Removed: {num_rows_removed}')

        return data_filtered

    @staticmethod
    def calculate_and_remove_null_values(df, groupby_column='Proteins', verbose=False): #TODO calcular y eliminar valores nulos.
        """
        Función para calcular y opcionalmente visualizar y eliminar filas con valores nulos en un DataFrame.

        Args:
        df (pd.DataFrame): DataFrame a analizar.
        groupby_column (str): Nombre de la columna para agrupar los resultados.
        verbose (bool): Si es True, imprime información y muestra gráficos.

        Returns:
        pd.DataFrame: DataFrame con filas nulas eliminadas.
        """
        # Crear una copia del DataFrame para no modificar el original
        temp_df = df.copy()

        # Calcular la cantidad de valores nulos en cada fila
        temp_df["null_count"] = temp_df.isnull().sum(axis=1)

        # Filtrar las filas que tienen al menos un valor nulo
        df_with_nulls = temp_df[temp_df["null_count"] > 0]

        # Información sobre los valores nulos
        num_rows_with_nulls = len(df_with_nulls)
        total_rows = len(df)
        proportion_nulls = num_rows_with_nulls / total_rows * 100

        # Eliminar filas con valores nulos
        data_filtered = temp_df[temp_df["null_count"] == 0]

        if verbose:
            # Gráfico
            plt.figure(figsize=(10, 4))
            sns.histplot(temp_df['null_count'], bins=range(1, temp_df['null_count'].max() + 1), kde=False)
            plt.title(f'Null Value Counts in {groupby_column}')
            plt.ylabel('Count')
            plt.xlabel('Number of Null Values')

            # Mostrar información de valores nulos
            print(f'{groupby_column} - Proportion of Rows with Nulls: {proportion_nulls:.2f}%')
            print(f'{groupby_column} - Number of Rows with Nulls: {num_rows_with_nulls}')
            num_rows_removed = total_rows - len(data_filtered)
            print(f'{groupby_column} - Number of Rows Removed: {num_rows_removed}')

        return data_filtered
    
    @staticmethod
    def remove_outliers_iqr(df, columns = ['NPX', 'PeptideAbundance'], iqr_factor=1.5):
        """
        Reemplaza los outliers basados en el Rango Intercuartílico (IQR) por los límites del IQR.
        """

        for column in columns:
            Q1 = df[column].quantile(0.25)
            Q3 = df[column].quantile(0.75)
            IQR = Q3 - Q1

            rango_inferior = Q1 - iqr_factor * IQR
            rango_superior = Q3 + iqr_factor * IQR

            # Reemplazar outliers por el rango inferior o superior
            df[column] = np.where(df[column] < rango_inferior, rango_inferior, df[column])
            df[column] = np.where(df[column] > rango_superior, rango_superior, df[column])

        return df

    
    @staticmethod #TODO
    def add_log_columns(df, columns=['NPX', 'PeptideAbundance']):

        """Añade columnas logarítmicas para las columnas especificadas."""

        for column in columns:
            df[f'{column}_log'] = np.log(df[column])
        return df

    @staticmethod
    def remove_outliers_std(df, columns=['NPX', 'PeptideAbundance'], std_factor=3):
        """
        Reemplaza los outliers en las columnas especificadas con NaN, basándose en la desviación estándar.
        """

        for column in columns:
            mean = df[column].mean()
            std = df[column].std()

            lower_bound = mean - std_factor * std
            upper_bound = mean + std_factor * std

            # Reemplazar outliers con NaN en lugar de eliminar filas
            df[column] = df[column].apply(lambda x: x if lower_bound <= x <= upper_bound else np.nan)

        return df
    
    @staticmethod #TODO
    def drop_upd23b_clinical_state_on_medication(df):
        df_transformed = df.drop(['upd23b_clinical_state_on_medication'], axis=1)
        return df_transformed

    @staticmethod #TODO
    def drop_group_key(df):
        df_transformed = df.drop(['group_key'], axis=1)
        return df_transformed
    
    @staticmethod #TODO
    def drop_null_count(df):
        df_transformed = df.drop(['null_count'], axis=1)
        return df_transformed
    
    @staticmethod #TODO
    def drop_visit_id_and_visit_month(df):
        df_transformed = df.drop(['visit_id', 'visit_month'], axis=1).reset_index(drop=True)
        return df_transformed
    
def full_EdaNew(dict_of_dfs : Dict[str, pd.DataFrame], month_to_filter : int = 24):

    """
    This function pretend to use the class EdaNew to orchestrate basic changes that involves the EDA.
    """

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

    return df_merged_all_filtered_full_Eda