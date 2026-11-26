
from pathlib import Path
import hashlib

import mlflow
import mlflow.sklearn
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.compose import ColumnTransformer
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from sklearn.pipeline import Pipeline

from mp081.cleaning import (
    load_clean_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)

ROOT_DIR = Path(__file__).resolve().parents[2]
TRAIN_PATH = ROOT_DIR / "data" / "raw" / "train.csv"

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "MP081-Energy-Prediction"
MODEL_NAME = "MP081-Energy-Model"


def calculate_data_hash():
    """Calculate SHA-256 from the untouched original training CSV."""
    sha256 = hashlib.sha256()

    with open(TRAIN_PATH, "rb") as file:
        for chunk in iter(lambda: file.read(8192), b""):
            sha256.update(chunk)

    return sha256.hexdigest()


def create_pipeline(n_estimators, max_depth):
    """Create preprocessing and regression pipeline."""
    preprocessing = ColumnTransformer(
        transformers=[
            ("numeric", "passthrough", NUMERIC_FEATURES),
            (
                "categorical",
                OneHotEncoder(handle_unknown="ignore"),
                CATEGORICAL_FEATURES,
            ),
        ]
    )

    model = RandomForestRegressor(
        n_estimators=n_estimators,
        max_depth=max_depth,
        random_state=42,
        n_jobs=-1,
    )

    return Pipeline([
        ("preprocessing", preprocessing),
        ("model", model),
    ])


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()
    data_hash = calculate_data_hash()

    train, _ = load_clean_data()

    X = train[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y = train["energy_kwh"]

    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
    )

    # Three configurations to compare.
    configurations = [
        {"n_estimators": 100, "max_depth": 10},
        {"n_estimators": 200, "max_depth": 15},
        {"n_estimators": 300, "max_depth": None},
    ]

    best_mae = float("inf")
    best_run_id = None
    best_version = None

    print("Starting MLflow training...")
    print("Original train.csv SHA-256:", data_hash)

    for config in configurations:
        pipeline = create_pipeline(
            config["n_estimators"],
            config["max_depth"],
        )

        with mlflow.start_run(
            run_name=f"RF_{config['n_estimators']}_{config['max_depth']}"
        ) as run:

            pipeline.fit(X_train, y_train)
            predictions = pipeline.predict(X_test)

            mae = mean_absolute_error(y_test, predictions)

            mlflow.log_param("data_sha256", data_hash)
            mlflow.log_param("n_estimators", config["n_estimators"])
            mlflow.log_param("max_depth", str(config["max_depth"]))
            mlflow.log_param("test_size", 0.20)
            mlflow.log_metric("mae", mae)

            model_info = mlflow.sklearn.log_model(
                sk_model=pipeline,
                name="model",
                registered_model_name=MODEL_NAME,
                serialization_format="cloudpickle",
            )

            version = model_info.registered_model_version

            print(
                f"Run: {run.info.run_id} | "
                f"MAE: {mae:.4f} | "
                f"Version: {version}"
            )

            if mae < best_mae:
                best_mae = mae
                best_run_id = run.info.run_id
                best_version = version

    # Give the best model the required champion alias.
    client.set_registered_model_alias(
        MODEL_NAME,
        "champion",
        str(best_version),
    )

    print("\n=== V1 TRAINING COMPLETE ===")
    print("Experiment:", EXPERIMENT_NAME)
    print("Best run ID:", best_run_id)
    print("Best MAE:", round(best_mae, 4))
    print("Registered model:", MODEL_NAME)
    print("Champion version:", best_version)


if __name__ == "__main__":
    main()
