import pandas as pd
import json


def load_data(file_path, timestep_interval):
    df = pd.read_csv(file_path, sep='\t')
    df['Timestep'] = pd.to_numeric(df['Timestep'], errors='coerce')
    df_selected_timesteps = df[df['Timestep'] % timestep_interval == 0]
    return df_selected_timesteps


def load_data_json(ruta_json):
    with open(ruta_json, 'r') as file:
        config = json.load(file)
    return config
