import os
import pandas as pd


def union(output_folder, csv_simulation):

    dataframes = []

    for archivo in os.listdir(output_folder):
        if archivo.endswith('.csv'):
            df = pd.read_csv(os.path.join(output_folder, archivo))
            dataframes.append(df)

    df_unido = pd.concat(dataframes, ignore_index=True)

    df_unido.to_csv(csv_simulation, index=False)
