import pickle
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

def normalize_dataframe(data, directory):
    if isinstance(data, str):
        df = pd.read_csv(data, sep=',')
        train= True
    elif isinstance(data, pd.DataFrame):
        df = data
        train = False

    targets = [col for col in df.columns if col.startswith('Target')]
    cols_to_exclude = ['Timestep', 'Sector'] + targets
    cols_to_normalize = df.columns.difference(cols_to_exclude)

    if train:
        scaler = MinMaxScaler()
        df[cols_to_normalize] = scaler.fit_transform(df[cols_to_normalize])
        pickle.dump(scaler, open('files/data/scaler.pkl', 'wb'))

        for target in targets:
            scaler_target = MinMaxScaler()
            df[[target]] = scaler_target.fit_transform(df[[target]])
            pickle.dump(scaler_target, open(f'files/data/scaler_{target}.pkl', 'wb'))

    else:
        scaler = pickle.load(open(directory+'\\files\\data\\scaler.pkl', 'rb'))
        df[cols_to_normalize] = scaler.transform(df[cols_to_normalize])

        for target in targets:
            scaler_target = pickle.load(open(directory+f'\\files\\data\\scaler_{target}.pkl', 'rb'))
            df[[target]] = scaler_target.transform(df[[target]])

    return df

def denormalize_predictions(predictions,target, directory):
    scaler = pickle.load(open(directory+f'\\files\\data\\scaler_{target}.pkl', 'rb'))
    temp_df = pd.DataFrame(predictions, columns=['Prediction'])
    denormalized_values = scaler.inverse_transform(temp_df)
    denormalized_df = pd.DataFrame(denormalized_values, columns=['Prediction'])
    return denormalized_df