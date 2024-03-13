import numpy as np
import pandas as pd


def _calculate_volume(environment):
    width = environment['width']
    height = environment['height']
    length = environment['length']
    return width * height * length


def _calculate_volumes(entities):
    volumes = {}
    for entity in entities:
        entity_name = entity.get('cellName', entity.get('name'))
        entity_radius = entity['radius']
        volumes[entity_name] = 4 / 3 * np.pi * (entity_radius ** 3)
    return volumes

def _calculate_diffusion_rates(molecules):
    diffusion_rates = {}
    for molecule in molecules:
        molecule_name = molecule['name']
        diffusion_rates[molecule_name] = molecule['diffusionRate']['exterior']
    return diffusion_rates

def preprocessing_data(df, config, n_divisions):
    total_volume = _calculate_volume(config['environment'])
    sector_volume = total_volume / (n_divisions ** 3)
    cell_volumes = _calculate_volumes(config['cells'])
    molecule_volumes = _calculate_volumes(config['agents']['molecules'])

    molecule_diffusion_rates = _calculate_diffusion_rates(config['agents']['molecules'])

    # Inicializa resultados
    results = []

    # Define límites de división en cada eje
    limits = {
        'X': np.linspace(df['X'].min(), df['X'].max(), n_divisions + 1),
        'Y': np.linspace(df['Y'].min(), df['Y'].max(), n_divisions + 1),
        'Z': np.linspace(df['Z'].min(), df['Z'].max(), n_divisions + 1)
    }

    for timestep in df['Timestep'].unique():
        df_timestep = df[df['Timestep'] == timestep]
        sector = 0

        for i in range(n_divisions):
            for j in range(n_divisions):
                for k in range(n_divisions):
                    OccupiedSpace = 0
                    df_sector = df_timestep[
                        (df_timestep['X'] >= limits['X'][i]) & (df_timestep['X'] < limits['X'][i + 1]) &
                        (df_timestep['Y'] >= limits['Y'][j]) & (df_timestep['Y'] < limits['Y'][j + 1]) &
                        (df_timestep['Z'] >= limits['Z'][k]) & (df_timestep['Z'] < limits['Z'][k + 1])
                        ]

                    sector_data = {
                        'Timestep': timestep,
                        'Sector': sector,
                    }


                    for name, volume in {**cell_volumes, **molecule_volumes}.items():
                        num_entities = len(df_sector[df_sector['Name'] == name])
                        sector_data[f'Num {name}'] = num_entities
                        sector_data[f'OccupiedSpace {name}'] = num_entities * volume
                        OccupiedSpace += num_entities * volume
                        if name in molecule_diffusion_rates:
                            sector_data[f'Diffusion Rate {name}'] = molecule_diffusion_rates[name]


                    sector_data['EmptySpace Sector'] = total_volume / n_divisions ** 3 - OccupiedSpace

                    results.append(sector_data)
                    sector = sector + 1

    return pd.DataFrame(results)
