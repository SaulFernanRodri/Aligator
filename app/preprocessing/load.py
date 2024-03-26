import pandas as pd
import json


def load_data(file_path):
    return pd.read_csv(file_path, sep='\t')


def load_data_json(ruta_json):
    with open(ruta_json, 'r') as file:
        config = json.load(file)
    return config
