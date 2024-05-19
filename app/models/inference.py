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


def save_predictions(predictions, output_path):
    predictions_df = pd.DataFrame(predictions, columns=['predicted_values'])
    predictions_df.to_csv(output_path, index=False)
    predictions_df.to_json(output_path, orient='records', lines=True)
