import pandas as pd
import json


def load_data(file_path, timestep_interval, chunk_size=100000):
    def process_chunk(chunk):
        chunk['Timestep'] = pd.to_numeric(chunk['Timestep'], errors='coerce')
        return chunk[chunk['Timestep'] % timestep_interval == 0]

    chunks = pd.read_csv(file_path, sep='\t', chunksize=chunk_size)
    df_selected_timesteps = pd.concat(process_chunk(chunk) for chunk in chunks)

    return df_selected_timesteps

def load_data_json(ruta_json):
    with open(ruta_json, 'r') as file:
        config = json.load(file)
    return config
