
from pathlib import Path

import mlflow
import mlflow.sklearn
import pandas as pd

from mp081.cleaning import clean_data, NUMERIC_FEATURES, CATEGORICAL_FEATURES

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
OUTPUT_DIR = ROOT_DIR / "predictions"

TRACKING_URI = "sqlite:///mlflow.db"
MODEL_NAME = "MP081-Energy-Model"


def main():
    mlflow.set_tracking_uri(TRACKING_URI)

    # Load the untouched holdout features.
    holdout = pd.read_csv(DATA_DIR / "holdout_features.csv")

    # Clean a copy using the same preprocessing rules.
    clean_holdout = clean_data(holdout)

    # Load the currently promoted model.
    model = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    # Use exactly the eight API/model features.
    features = NUMERIC_FEATURES + CATEGORICAL_FEATURES
    X = clean_holdout[features]

    predictions = model.predict(X)

    # Required output format.
    result = pd.DataFrame({
        "record_id": holdout["record_id"],
        "prediction": predictions,
    })

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    output_file = OUTPUT_DIR / "holdout_predictions.csv"
    result.to_csv(output_file, index=False)

    print("Champion model:", MODEL_NAME)
    print("Number of predictions:", len(result))
    print("Output columns:", list(result.columns))
    print("Missing predictions:", result["prediction"].isna().sum())
    print("Saved to:", output_file)
    print("\nFirst five predictions:")
    print(result.head().to_string(index=False))


if __name__ == "__main__":
    main()
