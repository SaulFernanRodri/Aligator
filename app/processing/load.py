import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_dataframe(data):
    if isinstance(data, str):
        df = pd.read_csv(data, sep=',')
        train= True
    elif isinstance(data, pd.DataFrame):
        df = data
        train = False
    cols_to_exclude = ['Timestep', 'Sector']
    cols_to_normalize = df.columns.difference(cols_to_exclude)
    scaler_path = 'files/scaler.pkl'
    if train:
        scaler = MinMaxScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        pickle.dump(scaler, open(scaler_path, 'wb'))
    else:
        scaler = pickle.load(open(scaler_path, 'rb'))
        df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])

    return df


def denormalize_predictions(predictions):
    scaler = pickle.load(open('files/scaler.pkl', 'rb'))
    temp_df = pd.DataFrame(predictions, columns=['Prediction'])
    denormalized_values = scaler.inverse_transform(temp_df)
    denormalized_df = pd.DataFrame(denormalized_values, columns=['Prediction'])
    return denormalized_df