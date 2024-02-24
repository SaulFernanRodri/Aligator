import argparse
from sklearn.model_selection import train_test_split
from processing.load import load_data
from preprocessing.load import load_data_preprocessing
from preprocessing.preprocessing import (normalize_data, define_and_assign_sectors, summarize_data,
                                         preprocessing_data, track_movements)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="Select a script option")
    parser.add_argument("-r", "--route", type=str, help="Route of the dataset", required=True)
    parser.add_argument("-c", "--number_clusters", type=int, default=4, help="Number of sectors(clusters)")
    args = parser.parse_args()

    # Model Parameters
    route = args.route
    n_clusters = args.number_clusters
    option = args.option

    # Global variables
    model_route = f"data/model"
    csv_train_route = f"data/train.csv"
    csv_test_route = f"data/test.csv"
    train_pickle_route = f"data/train.pkl"
    test_pickle_route = f"data/test.pkl"

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    if option == "preprocessing":
        # python app\main.py -o preprocessing -r "C:\Users\Saul\Desktop\TFG\pathogenic interactions\results\PCQuorum-SmSmX10_data_1708554589586_.txt" -c 4
        df = load_data_preprocessing(route)
        df = normalize_data(df)
        df, kmeans_model = define_and_assign_sectors(df, n_clusters)
        movements = track_movements(df, kmeans_model)
        preprocessed_df = summarize_data(df)
        simulation_df = preprocessing_data(preprocessed_df, movements)

        train_df, test_df = train_test_split(simulation_df, test_size=0.2, random_state=42)
        train_df.to_csv(csv_train_route, index=False)
        test_df.to_csv(csv_test_route, index=False)

    dataset_train, dataset_test = load_data(csv_train_route, csv_test_route, train_pickle_route, test_pickle_route)


if __name__ == '__main__':
    main()
