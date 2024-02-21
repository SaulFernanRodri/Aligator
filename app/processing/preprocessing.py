import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler


def normalize_data(df):
    # Normalizar las columnas de coordenadas, no es necesario
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
    # Resumir los datos por Timestep y Sector
    summary = df.groupby(['Timestep', 'Sector']).agg(
        alive=('StillAlive', lambda x: x.sum()),
        dead=('StillAlive', lambda x: (~x).sum())
    ).unstack(fill_value=0).stack()
    # Ajustar el formato de resumen
    summary_format = summary.reset_index().pivot_table(
        index='Timestep',
        columns='Sector',
        values=['alive', 'dead'],
        fill_value=0
    )
    summary_format.columns = ['{}_sector {}'.format(*col) for col in summary_format.columns]
    return summary_format.reset_index()


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
