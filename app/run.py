import argparse
import glob
import os
from processing import normalize_dataframe, denormalize_predictions
from preprocessing import load_data, load_data_json, preprocessing_data
from models import select_model, predict, save_predictions


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-j", "--route_json", type=str, help="Route of the json", required=True)
    args = parser.parse_args()

    route_json = args.route_json

    config=load_data_json(route_json)

    # Model Parameters desde el JSON
    folder_path = config["ml"]["mlOutput"]
    n_division = config["ml"]["parameter"]["division"]
    timesteps = config["ml"]["parameter"]["ts"]
    model_path = config["ml"]["parameter"]["model"]
    result_path = config["ml"]["parameter"]["results"]
    target = config["ml"]["parameter"]["tg"]

    model = select_model(model_path)

    filename = glob.glob(os.path.join(folder_path, "*.txt"))

    df = load_data(filename[0], 1)
    df.to_csv(folder_path+"df1.csv", index=False)

    simulation_df = preprocessing_data(df, config, n_division, 0, folder_path)
    simulation_df.to_csv(folder_path+"df.csv", index=False)
    simulation_df_normalize = normalize_dataframe("",simulation_df)

    model_features = model.feature_names_in_
    simulation_df_normalize = simulation_df_normalize[model_features]

    predictions = predict(model, simulation_df_normalize)

    predictions = denormalize_predictions(predictions)

    save_predictions(predictions, result_path)


if __name__ == '__main__':
    run()
