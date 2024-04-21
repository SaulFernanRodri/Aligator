import argparse
import os
import pickle
from processing import normalize_dataframe
from preprocessing import load_data, load_data_json, preprocessing_data
from models import traingbr, trainsvr, trainrandomforest, test, modeling


def main():
    # cd desktop\tfg\BioSpective\venv\scripts
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="Select a script option")
    parser.add_argument("-r", "--route_df", type=str, help="Route of the dataset")
    parser.add_argument("-j", "--route_json", type=str, help="Route of the json")
    parser.add_argument("-c", "--number_division", type=int, default=4,
                        help="Number of divisions, Example 2 divisions = 8 small cubes")
    parser.add_argument("-n", "--name", type=str, help="Name")
    args = parser.parse_args()

    # Model Parameters
    folder_path = args.route_df
    route_json = args.route_json
    n_division = args.number_division
    option = args.option
    name = args.name
    output_folder = r"C:\Users\Saul\Desktop\TFG\BioSpective\basura"

    # Global variables
    csv_simulation = f"datasets/simulation_{name}.csv"
    csv_simulation_normalize = f"datasets/simulation_moramlize_{name}.csv"

    if option == "preprocessing":
        # python app\main.py -o preprocessing -r "C:\Users\Saul\Desktop\TFG\pathogenic interactions\data" -j "C:\Users\Saul\Desktop\TFG\pathogenic interactions\inputs\Singulator - PCQuorum_1Sm1SmX10_peptide.json" -c 2
        for filename in os.listdir(folder_path):
            if filename.endswith(".txt"):  # Asegúrate de que el archivo es del tipo correcto
                route_df = os.path.join(folder_path, filename)

                df = load_data(route_df)
                config = load_data_json(route_json)

                simulation_df = preprocessing_data(df, config, n_division, 10)

                output_filename = filename.replace(".txt", "_processed.csv")
                simulation_df.to_csv(os.path.join(output_folder, output_filename), index=False)

    if option == "train":
        # Normalize the simulation data
        simulation_df = normalize_dataframe(csv_simulation)
        simulation_df.to_csv(csv_simulation_normalize, index=False)

        # Get the targets and the training, validation, and test data for each target
        targets, results = modeling(simulation_df)

        # For each target, train the random forest, gradient boosting, and support vector regression models
        # and save the models and the data to pickle files
        for target, data in results.items():
            rf = trainrandomforest(data['x_train'], data['y_train'], data['x_val'], data['y_val'])
            gbr = traingbr(data['x_train'], data['y_train'], data['x_val'], data['y_val'])
            svr = trainsvr(data['x_train'], data['y_train'], data['x_val'], data['y_val'])

            pickle.dump(rf, open(f"data/rf_{target}.pkl", 'wb'))
            pickle.dump(gbr, open(f"data/gbr_{target}.pkl", 'wb'))
            pickle.dump(svr, open(f"data/svr_{target}.pkl", 'wb'))

            pickle.dump((data['x_train'], data['y_train']), open(f"data/train_{target}.pkl", "wb"))
            pickle.dump((data['x_val'], data['y_val']), open(f"data/val_{target}.pkl", "wb"))
            pickle.dump((data['x_test'], data['y_test']), open(f"data/test_{target}.pkl", "wb"))

        # Save the target names
        with open('data/targets.pkl', 'wb') as f:
            pickle.dump(targets, f)

    if option == "test":
        # Load the target names
        with open('data/targets.pkl', 'rb') as f:
            targets = pickle.load(f)

        models = ['rf', 'gbr', 'svr']

        # For each target and each model, load the model and the test data from the pickle files
        # and test the model
        for target in targets:
            for model_name in models:
                model = pickle.load(open(f"data/{model_name}_{target}.pkl", 'rb'))

                x_test, y_test = pickle.load(open(f"data/test_{target}.pkl", "rb"))

                print(f"Testing {model_name} model for {target}")
                test(model, x_test, y_test)


if __name__ == '__main__':
    main()