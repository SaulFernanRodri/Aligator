import argparse
import os
import pickle
from processing import normalize_dataframe
from preprocessing import load_data, load_data_json, preprocessing_data, date_diff_in_seconds
from models import traingbr, trainsvr, trainrandomforest, test, modeling
from union import union


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
    output_folder = f"C:/Users/Saul/Desktop/TFG/BioSpective/preprocesing/{name}/{timesteps}/"
    csv_simulation = f"datasets/{name}/{route_csv}"
    csv_simulation_normalize = f"datasets/{name}/normalize_{route_csv}"
    results_folder = f"datasets/{name}"

    if option == "preprocessing":
        # python app\main.py -o preprocessing -r "C:\Users\Saul\Desktop\TFG\pathogenic interactions\data\data_peptide10"
        # -j "C:\Users\Saul\Desktop\TFG\pathogenic interactions\inputs\Singulator - PCQuorum_1Sm1SmX10_peptide.json"
        # -c 2 -n peptide_10 -ts 50 -csv "dataset_peptide_10_50.csv"
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):
                route_df = os.path.join(folder_path, filename)

                df = load_data(route_df, timesteps)
                config = load_data_json(route_json)

                simulation_df = preprocessing_data(df, config, n_division, timesteps, output_folder)

                output_filename = filename.replace(".txt", "_processed.csv")
                simulation_df.to_csv(os.path.join(output_folder, output_filename), index=False)
                date_diff_in_seconds(route_df, df, output_folder)

    union(output_folder, csv_simulation)

    if option == "train":
        # python app\main.py -o train -csv "dataset_peptide_10_500.csv" -n peptide_10 -ts 500
        simulation_df = normalize_dataframe(csv_simulation)
        simulation_df.to_csv(csv_simulation_normalize, index=False)

        targets, results = modeling(simulation_df)

        for target, data in results.items():
            pickle.dump((data['x_train'], data['y_train']), open(f"data/train_{target}.pkl", "wb"))
            pickle.dump((data['x_val'], data['y_val']), open(f"data/val_{target}.pkl", "wb"))
            pickle.dump((data['x_test'], data['y_test']), open(f"data/test_{target}.pkl", "wb"))

            pickle.dump(targets,  open('data/targets.pkl', 'wb'))

            rf = trainrandomforest(data['x_train'], data['y_train'], data['x_val'], data['y_val'], target,
                                   results_folder, timesteps)
            pickle.dump(rf, open(f"model/rf_{target}.pkl", 'wb'))

            gbr = traingbr(data['x_train'], data['y_train'], data['x_val'], data['y_val'], target, results_folder,
                           timesteps)
            pickle.dump(gbr, open(f"model/gbr_{target}.pkl", 'wb'))

            svr = trainsvr(data['x_train'], data['y_train'], data['x_val'], data['y_val'], target, results_folder,
                           timesteps)
            pickle.dump(svr, open(f"model/svr_{target}.pkl", 'wb'))

    if option == "test":

        targets = pickle.load(open('data/targets.pkl', 'rb'))

        models = ['rf', 'gbr', 'svr']

        for target in targets:
            for model_name in models:
                model = pickle.load(open(f"data/{model_name}_{target}.pkl", 'rb'))

                x_test, y_test = pickle.load(open(f"data/test_{target}.pkl", "rb"))

                print(f"Testing {model_name} model for {target}")
                test(model, x_test, y_test)


if __name__ == '__main__':
    main()
