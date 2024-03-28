import os
import pandas as pd

# Ruta de la carpeta donde están tus archivos CSV
ruta_carpeta_csv = 'C:\\Users\\Saul\\Desktop\\TFG\\BioSpective\\datasets'

# Lista para guardar los DataFrames de cada archivo CSV
dataframes = []

# Recorrer todos los archivos en la carpeta especificada
for archivo in os.listdir(ruta_carpeta_csv):
    if archivo.endswith('.csv'):
        # Construir la ruta completa al archivo
        ruta_completa = os.path.join(ruta_carpeta_csv, archivo)
        # Leer el archivo CSV y añadirlo a la lista de DataFrames
        df = pd.read_csv(ruta_completa)
        dataframes.append(df)

# Concatenar todos los DataFrames en uno solo
df_unido = pd.concat(dataframes, ignore_index=True)

# Ruta del archivo unido, cambia 'archivo_unido.csv' al nombre que prefieras
ruta_archivo_unido = 'C:\\Users\\Saul\\Desktop\\TFG\\BioSpective\\datasets\\simulation_todos.csv'

# Guardar el DataFrame unido en un nuevo archivo CSV
df_unido.to_csv(ruta_archivo_unido, index=False)

print(f'Archivo unido creado en: {ruta_archivo_unido}')
