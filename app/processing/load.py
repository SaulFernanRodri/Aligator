import pandas as pd


def load_data(file_path):
    # Cargar el archivo delimitado por tabulaciones
    return pd.read_csv(file_path, sep='\t')

