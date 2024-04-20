import os
import pandas as pd

ruta_carpeta_csv = 'C:\\Users\\Saul\\Desktop\\TFG\\BioSpective\\datasets'

dataframes = []


for archivo in os.listdir(ruta_carpeta_csv):
    if archivo.endswith('.csv'):
        ruta_completa = os.path.join(ruta_carpeta_csv, archivo)
        df = pd.read_csv(ruta_completa)
        dataframes.append(df)

df_unido = pd.concat(dataframes, ignore_index=True)


ruta_archivo_unido = 'C:\\Users\\Saul\\Desktop\\TFG\\BioSpective\\datasets\\simulation_todos.csv'


df_unido.to_csv(ruta_archivo_unido, index=False)

print(f'Archivo unido creado en: {ruta_archivo_unido}')
