import argparse
from sklearn.model_selection import train_test_split
from processing.load import load_data
from preprocessing.load import load_data_preprocessing, load_data_json
from preprocessing.preprocessing import (preprocessing_data)


def main():
    #cd desktop\tfg\tfg_saul\venv\scripts
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="Select a script option")
    parser.add_argument("-r", "--route_df", type=str, help="Route of the dataset", required=True)
    parser.add_argument("-j", "--route_json", type=str, help="Route of the json", required=True)
    parser.add_argument("-c", "--number_division", type=int, default=4, help="Number of divisions")
    args = parser.parse_args()

    # Model Parameters
    route_df = args.route_df
    route_json = args.route_json
    n_division = args.number_division
    option = args.option

    # Global variables
    model_route = f"data/model"
    csv_simulation = f"data/simulation.csv"
    csv_train_route = f"data/train.csv"
    csv_test_route = f"data/test.csv"
    train_pickle_route = f"data/train.pkl"
    test_pickle_route = f"data/test.pkl"

    x_train = []
    y_train = []
    x_test = []
    y_test = []

    if option == "preprocessing":
        # python app\main.py -o preprocessing -r "C:\Users\Saul\Desktop\TFG\pathogenic interactions\results\PCQuorum-SmSmX10_data_1709759337212_.txt" -j "C:\Users\Saul\Desktop\TFG\pathogenic interactions\inputs\Singulator - PCQuorum_1Sm1SmX10_peptide.json" -c 4
        df = load_data_preprocessing(route_df)
        config = load_data_json(route_json)
        simulation_df = preprocessing_data(df, config, n_division)
        simulation_df.to_csv(csv_simulation, index=False)
        train_df, test_df = train_test_split(simulation_df, test_size=0.2, random_state=42)
        train_df.to_csv(csv_train_route, index=False)
        test_df.to_csv(csv_test_route, index=False)

    dataset_train, dataset_test = load_data(csv_train_route, csv_test_route, train_pickle_route, test_pickle_route)


if __name__ == '__main__':
    main()
