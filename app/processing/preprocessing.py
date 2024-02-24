import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def normalize_data(df):
    scaler = StandardScaler()
    df[['X', 'Y', 'Z']] = scaler.fit_transform(df[['X', 'Y', 'Z']])
    return df


def define_and_assign_sectors(df, n_clusters=4):
    # Definir sectores usando el primer timestep
    initial_timestep_df = df[df['Timestep'] == 0]
    kmeans = KMeans(n_clusters=n_clusters, random_state=0)
    kmeans.fit(initial_timestep_df[['X', 'Y', 'Z']])

    # Asignar sectores en todos los timesteps usando los centros de cluster fijos
    cluster_centers = kmeans.cluster_centers_
    kmeans = KMeans(n_clusters=n_clusters, init=cluster_centers, n_init=1, max_iter=1)
    df['Sector'] = kmeans.fit_predict(df[['X', 'Y', 'Z']])

    return df, kmeans


def summarize_data(df):
    # Contar el número total de entidades (células y moléculas juntas) por Timestep y Sector
    summary = df.groupby(['Timestep', 'Sector']).size().reset_index(name='Entity_Count')

    # Convertir el resumen en un formato de tabla pivotante para facilitar la visualización
    summary_pivot = summary.pivot(index='Timestep', columns='Sector', values='Entity_Count').fillna(0)
    summary_pivot.columns = [f'entities_sector_{col}' for col in summary_pivot.columns]

    return summary_pivot.reset_index()


def integrate_movements_to_summary(df, movements):
    movements_pivot = movements.pivot_table(index='Timestep', columns=['Prev_Sector', 'Sector'], values='Moved_Cells',
                                            fill_value=0)

    # Aplanar el MultiIndex en las columnas
    movements_pivot.columns = ['mov_{}_to_{}'.format(int(from_sec), int(to_sec)) for from_sec, to_sec in
                               movements_pivot.columns]

    summary_df = pd.merge(df, movements_pivot, on='Timestep', how='left').fillna(0)

    return summary_df


def track_movements(df, kmeans_model):
    # Predecir sectores para el DataFrame completo
    df['Sector'] = kmeans_model.predict(df[['X', 'Y', 'Z']])
    df['Prev_Sector'] = df.groupby('ID')['Sector'].shift(1)
    df['Moved'] = df['Sector'] != df['Prev_Sector']

    # Crear un DataFrame para rastrear movimientos
    movements = pd.DataFrame()

    for ts in df['Timestep'].unique()[1:]:  # Empezamos desde el segundo timestep
        ts_df = df[df['Timestep'] == ts]
        # Aquí, en lugar de filtrar por 'Moved', directamente agrupamos por 'Prev_Sector' y 'Sector'
        movements_df = ts_df.groupby(['Prev_Sector', 'Sector']).size().reset_index(name='Moved_Cells')
        movements_df = movements_df[
            movements_df['Moved_Cells'] > 0]  # Filtrar movimientos donde al menos una célula se ha movido
        movements_df['Timestep'] = ts
        movements = pd.concat([movements, movements_df], ignore_index=True)

    return movements
