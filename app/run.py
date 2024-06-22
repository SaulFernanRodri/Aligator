import argparse
import glob
import os
import pandas as pd
from processing import normalize_dataframe, denormalize_predictions
from preprocessing import load_data, load_data_json, preprocessing_data
from models import select_model, predict, save_predictions


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--route_json", type=str, help="Route of the json", required=True)
    args = parser.parse_args()

    route_json = args.route_json

    config = load_data_json(route_json)

    # Model Parameters JSON
    directory = config["ml"]["python"]
    folder_path = config["ml"]["mlOutput"]
    n_division = config["ml"]["division"]
    models = config["ml"]["models"]
    targets = config["ml"]["targets"]

    if len(models) != len(targets):
        raise ValueError("La cantidad de modelos y targets debe ser igual")

    filename = glob.glob(os.path.join(folder_path, "*.txt"))

    df = load_data(filename[0], 1)
    for file in filename:
       os.remove(file)

    simulation_df = preprocessing_data(df, config, n_division, 0, folder_path)
    simulation_df.to_csv(folder_path + "preprocesado.csv", index=False)

    simulation_df_normalize = normalize_dataframe(simulation_df, directory)
    simulation_df_normalize.to_csv(folder_path + "normalizado.csv", index=False)

    final_dfs = []
    for model_path, target in zip(models, targets):
        model = select_model(model_path)

        model_features = model.feature_names_in_
        simulation_df_normalize_selected = simulation_df_normalize[model_features]

        predictions = predict(model, simulation_df_normalize_selected)
        predictions = denormalize_predictions(predictions, target, directory)
        predictions = pd.DataFrame(predictions).astype(int)

        final_df = pd.DataFrame()
        final_df['Sector'] = simulation_df['Sector'].values
        target_value = target.split()[1]
        final_df['Target'] = target_value
        final_df['Prediction'] = predictions.values

        final_dfs.append(final_df)

    final_df_combined = pd.concat(final_dfs, ignore_index=True)
    save_predictions(final_df_combined, folder_path)


if __name__ == '__main__':
    run()

