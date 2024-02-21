import argparse
from processing.load import load_data
from processing.preprocessing import normalize_data, define_and_assign_sectors, summarize_data, track_movements


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="Select a script option")
    parser.add_argument("-r", "--route", type=str, help="Route for training model")
    parser.add_argument("-c", "--number_clusters", type=int, default=4, help="Number of clusters")
    args = parser.parse_args()

    if args.option == "preprocessing":
        # python app\main.py -o preprocessing -r "C:\Users\Saul\Desktop\TFG\pathogenic interactions\results\PCQuorum-SmSmX10_data_1708554589586_.txt" -c 4
        df = load_data(args.route)
        df = normalize_data(df)
        df, kmeans_model = define_and_assign_sectors(df, n_clusters=args.number_clusters)
        preprocessed_data, movements_data = summarize_data(df), track_movements(df, kmeans_model)
        preprocessed_data.to_csv('files/preprocessed_data.csv', index=False)
        movements_data.to_csv('files/movements_data.csv', index=False)

    elif args.option == "train_model":
        print(f"This is the route: {args.route}")

    elif args.option == "test_model":
        print("Hello world prueba 2 visual")


if __name__ == '__main__':
    main()
