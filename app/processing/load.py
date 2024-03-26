import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_dataframe(file_path):
    df = pd.read_csv(file_path, sep=',')
    # Seleccionar columnas que no se quieren normalizar
    cols_to_exclude = ['Timestep', 'Sector']

    # Seleccionar las columnas que se van a normalizar
    cols_to_normalize = df.columns.difference(cols_to_exclude)

    # Inicializar el normalizador
    scaler = MinMaxScaler()

    # Normalizar los datos y ajustar al rango deseado
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    return df




