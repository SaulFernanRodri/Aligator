import argparse
from processing import read
from utils import create_sections


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="Select a script option")
    parser.add_argument("-r", "--route", type=str, help="Route for training model")
    parser.add_argument("-c", "--number_clusters", type=int, help="Number of clusters")
    args = parser.parse_args()

    if args.option == "preprocessing":
        #python app\main.py -o preprocessing -r "C:\Users\Saul\Desktop\TFG\pathogenic interactions\results\PCQuorum-SmSmX01_Displacement_1707565575355_.txt" -c 4

        create_sections(args.route, args.number_clusters)

    elif args.option == "train_model":
        print(f"This is the route: {args.route}")

    elif args.option == "test_model":
        print("Hello world prueba 2 visual")
        read(args.route)


if __name__ == '__main__':
    main()
