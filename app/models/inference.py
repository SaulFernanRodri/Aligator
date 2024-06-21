import joblib
import pandas as pd
import os


def select_model(model_path):
    if os.path.exists(model_path):
        return joblib.load(model_path)
    else:
        raise FileNotFoundError(f"Model not found at {model_path}.")


def predict(model, data):
    return model.predict(data)


def save_predictions(predictions, result_path):
    predictions.to_csv(result_path, index=False)

