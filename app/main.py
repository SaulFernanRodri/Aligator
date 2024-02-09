import argparse

if __name__ == '__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument("-o", "--option", type=str, help="Select an script option")
    parser.add_argument("-r", "--route", type=str, help="Route for training model")
    parser.add_argument("-g", "--gamma", type=str, help="Parameter gamma SVC")
    args = parser.parse_args()

    if args.option == "train_model":
        print(f"This is the route: {args.route}")

    if args.option == "test_model":
        # python app\training_scripts.py -o
        print("Hello world")
