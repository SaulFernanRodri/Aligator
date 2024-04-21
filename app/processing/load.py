import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler


def normalize_dataframe(file_path):
    df = pd.read_csv(file_path, sep=',')
    cols_to_exclude = ['Timestep', 'Sector']
    cols_to_normalize = df.columns.difference(cols_to_exclude)
    scaler = MinMaxScaler()
    df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
    pickle.dump(scaler, open('scaler.pkl', 'wb'))
    return df
