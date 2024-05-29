import argparse
from processing import normalize_dataframe
from preprocessing import load_data, load_data_json, preprocessing_data
from models import select_model, predict, save_predictions


def run():
    parser = argparse.ArgumentParser()
    parser.add_argument("-r", "--route_df", type=str, help="Route of the dataset")
    parser.add_argument("-j", "--route_json", type=str, help="Route of the json")
    parser.add_argument("-csv", "--route_csv", type=str, help="Route of the csv")
    parser.add_argument("-c", "--number_division", type=int, default=2,
                        help="Number of divisions, Example 2 divisions = 8 small cubes")
    parser.add_argument("-ts", "--timesteps", type=int, help="Timesteps interval")
    parser.add_argument("-tg", "--target", type=str, help="Target")
    parser.add_argument("-m", "--model", type=str, help="Model")
    parser.add_argument("-r", "--results", type=str, help="Results")
    args = parser.parse_args()

    # Model Parameters
    folder_path = args.route_df
    route_json = args.route_json
    n_division = args.number_division
    timesteps = args.timesteps
    model_path = args.model
    result_path = args.results

    model = select_model(model_path)

    df = load_data(folder_path, 0)

    config = load_data_json(route_json)

    simulation_df = preprocessing_data(df, config, n_division, timesteps, "")
    simulation_df = normalize_dataframe(simulation_df)

    predictions = predict(model, simulation_df)

    save_predictions(predictions, result_path)


if __name__ == '__run__':
    run()
