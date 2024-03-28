import argparse
import pickle
from processing import normalize_dataframe
from preprocessing import load_data, load_data_json, preprocessing_data
from models import trainGBR, trainSVR, trainRandomForest, test, model


def main():
    #cd desktop\tfg\BioSpective\venv\scripts
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="Select a script option")
    parser.add_argument("-r", "--route_df", type=str, help="Route of the dataset")
    parser.add_argument("-j", "--route_json", type=str, help="Route of the json")
    parser.add_argument("-c", "--number_division", type=int, default=4, help="Number of divisions, Example 2 divisions = 8 small cubes")
    parser.add_argument("-n", "--name", type=str, help="Name")
    args = parser.parse_args()

    # Model Parameters
    route_df = args.route_df
    route_json = args.route_json
    n_division = args.number_division
    option = args.option
    name = args.name

    # Global variables
    rf_route = f"data/ramdomForest.pkl"
    svr_route = f"data/SVR.pkl"
    gbr_route = f"data/GBR.pkl"
    csv_simulation = f"datasets/simulation_{name}.csv"
    csv_simulation_normalize = f"datasets/simulation_moramlize_{name}.csv"
    train_pickle_route = f"data/train.pkl"
    val_pickle_route = f"data/vaL.pkl"
    test_pickle_route = f"data/test.pkl"


    X_train = []
    y_train = []
    X_val = []
    y_val = []
    X_test = []
    y_test = []


    if option == "preprocessing":
        # python app\main.py -o preprocessing -r "C:\Users\Saul\Desktop\TFG\pathogenic interactions\data\PCQuorum-SmSmX10_data_1710490541632_.txt" -j "C:\Users\Saul\Desktop\TFG\pathogenic interactions\inputs\Singulator - PCQuorum_1Sm1SmX10_peptide.json" -c 2 -n Peptide_1
        # python app\main.py -o preprocessing -r "C:\Users\Saul\Desktop\TFG\pathogenic interactions\results\PCQuorum-SmPa5X_data_1710488974828_.txt" -j "C:\Users\Saul\Desktop\TFG\pathogenic interactions\inputs\muitas.json" -c 2 -n Muchas_2
        df = load_data(route_df)
        config = load_data_json(route_json)
        simulation_df = preprocessing_data(df, config, n_division,10)
        simulation_df.to_csv(csv_simulation, index=False)



    if option == "train":
        # python app\main.py -o train -n todos
        simulation_df = normalize_dataframe(csv_simulation)
        simulation_df.to_csv(csv_simulation_normalize, index=False)
        X_train, X_val, X_test, y_train, y_val, y_test = model(simulation_df)

        rf = trainRandomForest(X_train, y_train, X_val, y_val)
        gbr = trainGBR(X_train, y_train, X_val, y_val)
        svr = trainSVR(X_train, y_train, X_val, y_val)

        pickle.dump(rf, open(rf_route, 'wb'))
        pickle.dump(gbr, open(gbr_route, 'wb'))
        pickle.dump(svr, open(svr_route, 'wb'))

        pickle.dump((X_train, y_train), open(train_pickle_route, "wb"))
        pickle.dump((X_val, y_val), open(val_pickle_route, "wb"))
        pickle.dump((X_test, y_test), open( test_pickle_route, "wb"))

    if option == "test":
        rf = pickle.load(open(svr_route, 'rb'))

        X_test, y_test = pickle.load(open("data/test_data.pkl", "rb"))

        test(rf, X_test, y_test)


if __name__ == '__main__':
    main()
