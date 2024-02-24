import pandas as pd
import pickle
import os


def load_data(csv_train_route, csv_test_route, train_pickle_route, test_pickle_route):

    if not os.path.isfile(train_pickle_route):
        train_data = pd.read_csv(csv_train_route)
        with open(train_pickle_route, 'wb') as f:
            pickle.dump(train_data, f)
    else:
        with open(train_pickle_route, 'rb') as f:
            train_data = pickle.load(f)

    if not os.path.isfile(test_pickle_route):
        test_data = pd.read_csv(csv_test_route)
        with open(test_pickle_route, 'wb') as f:
            pickle.dump(test_data, f)
    else:
        with open(test_pickle_route, 'rb') as f:
            test_data = pickle.load(f)

    return train_data, test_data
