import pandas as pd
import os

HIPERPARAMETROS = {
    'mes_default' : 24,
    'mes_a_predecir' : 84,
    'semilla' : 42,
    'n_jobs' : 16,
    'tipo_de_model' : 'lightgbm',
    'modelos' : ['updrs_1', 'updrs_2', 'updrs_3', 'updrs_4'],
    # 'ruta_de_mejores_modelos_joblibs' : os.path.join(os.getcwd(), '..', 'data', 'outputs', 'pruebas', 'mejores_modelos_joblibs'),
    # 'ruta_de_mejores_modelos_parametros_json' : os.path.join(os.getcwd(), '..', 'data', 'outputs', 'pruebas', 'mejores_modelos_parametros_json'),
    # 'ruta_study_optuna' : os.path.join(os.getcwd(), '..', 'data', 'outputs', 'pruebas', 'study_optuna'),
    # 'ruta_guardar_imagenes' : os.path.join(os.getcwd(), '..', 'data', 'outputs', 'pruebas', 'graficas'),
    'ruta_guardar_metricas' : os.path.join(os.getcwd(), '..', 'data', 'outputs', 'pruebas', 'metricas')
    }

# Ir a la ruta donde se encuentran los csv de las metricas.
os.chdir(HIPERPARAMETROS['ruta_guardar_metricas'])

# Totos los csv que hay en la ruta transformarlos a tablas de latex.
# solo para los que acaben en .csv
# y al nombre final quitale la parte .csv y añade .tex
for file in os.listdir():
    if file.endswith('.csv'):
        df = pd.read_csv(file)
        df.to_latex(file[:-4] + '.tex', index=False)


