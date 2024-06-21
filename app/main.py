import argparse
import glob
import os
import pickle
import pandas as pd
from joblib import Parallel, delayed

from processing import normalize_dataframe
from preprocessing import load_and_preprocess, load_data_json
from models import execute_model, test, modeling


def main():
    # cd desktop\tfg\BioSpective\venv\scripts
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="Select a script option")
    parser.add_argument("-r", "--route_df", type=str, help="Route of the dataset")
    parser.add_argument("-j", "--route_json", type=str, help="Route of the json")
    parser.add_argument("-csv", "--route_csv", type=str, help="Route of the csv")
    parser.add_argument("-c", "--number_division", type=int, default=2,
                        help="Number of divisions, Example 2 divisions = 8 small cubes")
    parser.add_argument("-n", "--name", type=str, help="Name")
    parser.add_argument("-ts", "--timesteps", type=int, help="Timesteps interval")
    parser.add_argument("-tg", "--target", type=str, help="Target")
    args = parser.parse_args()

    # Model Parameters
    folder_path = args.route_df
    route_json = args.route_json
    route_csv = args.route_csv
    n_division = args.number_division
    option = args.option
    name = args.name
    timesteps = args.timesteps

    # Global variables
    output_folder = f"files/data_processed/{name}/{timesteps}/"
    csv_simulation = f"files/data_train/{name}/{route_csv}"
    csv_simulation_normalize = f"files/data_train/{name}/normalize_{route_csv}"
    results_folder = f"files/data_train/{name}"

    if option == "preprocessing":
        if option == "preprocessing":
            # Use glob to get all txt files
            txt_files = glob.glob(os.path.join(folder_path, "*.txt"))

            config = load_data_json(route_json)
            # Use joblib to parallelize the loading and preprocessing
            dataframes = Parallel(n_jobs=-1)(
                delayed(load_and_preprocess)(filename, config, timesteps, output_folder, n_division) for filename in txt_files)

            full_df = pd.concat(dataframes, ignore_index=True)
            full_df.to_csv(csv_simulation, index=False)

    if option == "train":
        # python app\main.py -o train -csv "dataset_peptide_10_500.csv" -n peptide_10 -ts 500
        simulation_df = normalize_dataframe(csv_simulation)
        simulation_df.to_csv(csv_simulation_normalize, index=False)

        targets, results = modeling(simulation_df)

        for target, data in results.items():
            pickle.dump((data['x_train'], data['y_train']), open(f"files/data/train_{target}.pkl", "wb"))
            pickle.dump((data['x_val'], data['y_val']), open(f"files/data/val_{target}.pkl", "wb"))

            execute_model(data['x_train'], data['y_train'], data['x_val'], data['y_val'],
                          target, results_folder, timesteps)

    if option == "test":
        targets = pickle.load(open('files/data/targets.pkl', 'rb'))

        models = ['rf', 'gbr', 'svr']

        for target in targets:
            for model_name in models:
                model = pickle.load(open(f"files/data/{model_name}_{target}.pkl", 'rb'))

                x_test, y_test = pickle.load(open(f"files/data/test_{target}.pkl", "rb"))

                print(f"Testing {model_name} model for {target}")
                test(model, x_test, y_test)


if __name__ == '__main__':
    main()
