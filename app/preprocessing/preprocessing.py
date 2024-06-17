import numpy as np
import pandas as pd
import os
from .load_pre import load_data


def _calculate_volume(environment):
    width = environment['width']
    height = environment['height']
    length = environment['length']
    return width, height, length


def _calculate_volumes(entities):
    volumes = {}
    for entity in entities:
        entity_name = entity.get('cellName', entity.get('name'))
        entity_radius = entity['radius']
        if 'height' in entity and entity['height'] != 0:
            entity_height = entity['height']
            volumes[entity_name] = np.pi * (entity_radius ** 2) * entity_height
        else:
            volumes[entity_name] = 4 / 3 * np.pi * (entity_radius ** 3)
    return volumes


def _calculate_diffusion_rates(molecules):
    diffusion_rates = {}
    for molecule in molecules:
        molecule_name = molecule['name']
        diffusion_rates[molecule_name] = molecule['diffusionRate']['exterior']
    return diffusion_rates

# limits['X'][i] y limits['X'][i + 1]: Son los límites en el eje X para el sector actual que se
# está analizando.
# limits['X'] es un array que contiene los puntos de inicio y fin de cada división en el eje X,
# i es el índice de la iteración actual en el bucle que recorre estas divisiones. limits['X'][i]
# es el límite inferior,
# y limits['X'][i + 1] es el límite superior del sector en el eje X.
# #(df_timestep['X'] >= limits['X'][i]) & (df_timestep['X'] < limits['X'][i + 1]):
# Esta condición selecciona solo las filas del DataFrame df_timestep donde el valor de la columna
# 'X' está dentro de los límites del sector actual en el eje X.


def _select_sector(df, limits, i, j, k):
    df_sector = df[
        (df['X'] >= limits['X'][i]) & (df['X'] < limits['X'][i + 1]) &
        (df['Y'] >= limits['Y'][j]) & (df['Y'] < limits['Y'][j + 1]) &
        (df['Z'] >= limits['Z'][k]) & (df['Z'] < limits['Z'][k + 1])
    ]
    return df_sector


def write_limits_to_file(limits, output_folder, output_filename='limits.csv'):
    n_divisions = len(limits['X']) - 1
    with open(os.path.join(output_folder, output_filename), 'w') as file:
        sector_count = 0
        for i in range(n_divisions):
            for j in range(n_divisions):
                for k in range(n_divisions):
                    iniX = limits["X"][i]
                    endX = limits["X"][i + 1]
                    iniY = limits["Y"][j]
                    endY = limits["Y"][j + 1]
                    iniZ = limits["Z"][k]
                    endZ = limits["Z"][k + 1]
                    # Escribir en el formato solicitado: primero sector, luego los límites en orden X, Y, Z
                    file.write(f'{sector_count}, {iniX}, {endX}, {iniY}, {endY}, {iniZ}, {endZ}\n')
                    sector_count += 1


def preprocessing_data(df, config, n_divisions, future_step, output_folder):
    width, height, length = _calculate_volume(config['environment'])
    total_volume = width * height * length
    cell_volumes = _calculate_volumes(config['cells'])
    molecule_volumes = _calculate_volumes(config['agents']['molecules'])

    molecule_diffusion_rates = _calculate_diffusion_rates(config['agents']['molecules'])

    results = []

    limits = {
        'X': np.linspace(0, width*1000, n_divisions + 1),
        'Y': np.linspace(0, height*1000, n_divisions + 1),
        'Z': np.linspace(0, length*1000, n_divisions + 1),
    }

    if output_folder != "":
        write_limits_to_file(limits, output_folder)

    # Por ejemplo, si df['X'].min() es 0, df['X'].max() es 10, y n_divisions es 2, entonces np.linspace(0, 10, 3)
    # generará [0, 5, 10].
    # Esto significa que el espacio en el eje X se dividirá en 2 sectores, con límites en 0, 5, y 10.

    targets = {}
    if(future_step != 0 ):
        for timestep in sorted(df['Timestep'].unique()):
            future_timestep = timestep + future_step
            df_future = df[df['Timestep'] == future_timestep]
            for i in range(n_divisions):
                for j in range(n_divisions):
                    for k in range(n_divisions):
                        df_future_sector = _select_sector(df_future, limits, i, j, k)
                        for name in {**cell_volumes, **molecule_volumes}.keys():
                            num_entities = len(df_future_sector[df_future_sector['Name'] == name])
                            targets[(timestep, i, j, k, name)] = num_entities

    for timestep in df['Timestep'].unique():
        df_timestep = df[df['Timestep'] == timestep]
        sector = 0

        for i in range(n_divisions):
            for j in range(n_divisions):
                for k in range(n_divisions):
                    occupiedspace = 0

                    df_sector = _select_sector(df_timestep, limits, i, j, k)

                    sector_data = {
                        'Timestep': timestep,
                        'Sector': sector,
                    }

                    # combinar dos diccionarios "unpacking operator" (**).
                    for name, volume in {**cell_volumes, **molecule_volumes}.items():
                        num_entities = len(df_sector[df_sector['Name'] == name])
                        sector_data[f'Num {name}'] = num_entities
                        sector_data[f'OccupiedSpace {name}'] = num_entities * volume
                        if name in molecule_diffusion_rates:
                            sector_data[f'Diffusion Rate {name}'] = molecule_diffusion_rates[name]
                        occupiedspace += num_entities * volume
                        key = (timestep, i, j, k, name)
                        sector_data[f'Target {name}'] = targets.get(key, 0)

                    sector_data['EmptySpace Sector'] = total_volume / n_divisions ** 3 - occupiedspace

                    results.append(sector_data)
                    sector = sector + 1

    return pd.DataFrame(results)


def date_diff_in_seconds(name, df, output_folder, date_column='Date', output_filename='time.txt'):

    df[date_column] = pd.to_datetime(df[date_column])
    timedelta = df[date_column].iloc[-1] - df[date_column].iloc[0]
    diff_seconds = timedelta.total_seconds()

    with open(os.path.join(output_folder, output_filename), 'a') as file:
        file.write(f'{name} : {diff_seconds}\n')


def load_and_preprocess(filename, config, timesteps, output_folder, n_division):
    df = load_data(filename, timesteps)
    simulation_df = preprocessing_data(df, config, n_division, timesteps, output_folder)
    date_diff_in_seconds(filename, df, output_folder)
    return simulation_df
