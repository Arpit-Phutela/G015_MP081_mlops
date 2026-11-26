
from pathlib import Path

import mlflow
import mlflow.sklearn
import numpy as np
import pandas as pd

from mlflow.tracking import MlflowClient
from sklearn.metrics import mean_absolute_error

from mp081.cleaning import (
    load_clean_data,
    NUMERIC_FEATURES,
    CATEGORICAL_FEATURES,
)
from mp081.train import create_pipeline, calculate_data_hash

ROOT_DIR = Path(__file__).resolve().parents[2]

TRACKING_URI = "sqlite:///mlflow.db"
EXPERIMENT_NAME = "MP081-Energy-Prediction"
MODEL_NAME = "MP081-Energy-Model"


def calculate_psi(reference, current, bins=10):
    """Calculate Population Stability Index using reference quantile bins."""
    reference = pd.Series(reference).dropna().astype(float)
    current = pd.Series(current).dropna().astype(float)

    edges = np.unique(
        np.quantile(reference, np.linspace(0, 1, bins + 1))
    )

    if len(edges) < 2:
        return 0.0

    edges[0] = -np.inf
    edges[-1] = np.inf

    ref_counts = np.histogram(reference, bins=edges)[0]
    cur_counts = np.histogram(current, bins=edges)[0]

    ref_pct = np.clip(ref_counts / len(reference), 0.0001, None)
    cur_pct = np.clip(cur_counts / len(current), 0.0001, None)

    return float(np.sum(
        (cur_pct - ref_pct) * np.log(cur_pct / ref_pct)
    ))


def main():
    mlflow.set_tracking_uri(TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)

    client = MlflowClient()

    train, later = load_clean_data()

    # 1. Load current champion v1.
    v1 = mlflow.sklearn.load_model(
        f"models:/{MODEL_NAME}@champion"
    )

    # 2. PSI: cleaned training data compared with cleaned later data.
    psi_results = {}

    for column in NUMERIC_FEATURES:
        psi_results[column] = calculate_psi(
            train[column],
            later[column],
            bins=10,
        )

    most_drifted = max(psi_results, key=psi_results.get)

    print("\n=== POPULATION STABILITY INDEX ===")
    for column, value in psi_results.items():
        print(f"{column}: {value:.4f}")

    print("Most drifted feature:", most_drifted)

    # 3. v1 MAE separately for each station type.
    X_later = later[NUMERIC_FEATURES + CATEGORICAL_FEATURES]
    y_later = later["energy_kwh"]

    v1_predictions = v1.predict(X_later)

    later = later.copy()
    later["v1_prediction"] = v1_predictions

    print("\n=== V1 MAE BY STATION TYPE ===")
    group_mae = {}

    for group, rows in later.groupby("station_type"):
        score = mean_absolute_error(
            rows["energy_kwh"],
            rows["v1_prediction"],
        )
        group_mae[group] = float(score)
        print(f"{group}: {score:.4f}")

    worst_group = max(group_mae, key=group_mae.get)
    print("Worst served group:", worst_group)

    # 4. Sort later data by date and split chronologically 70:30.
    later = later.sort_values("date").reset_index(drop=True)

    split_index = int(len(later) * 0.70)

    first_70 = later.iloc[:split_index].copy()
    last_30 = later.iloc[split_index:].copy()

    # 5. Evaluate v1 on the last 30%.
    check_features = NUMERIC_FEATURES + CATEGORICAL_FEATURES

    X_check = last_30[check_features]
    y_check = last_30["energy_kwh"]

    v1_check_predictions = v1.predict(X_check)
    mae_v1 = mean_absolute_error(y_check, v1_check_predictions)

    # 6. Train v2 on cleaned train + first 70% of later.
    combined = pd.concat(
        [train, first_70],
        ignore_index=True,
    )

    X_combined = combined[check_features]
    y_combined = combined["energy_kwh"]

    v2 = create_pipeline(
        n_estimators=300,
        max_depth=None,
    )

    v2.fit(X_combined, y_combined)

    v2_predictions = v2.predict(X_check)
    mae_v2 = mean_absolute_error(y_check, v2_predictions)

    # 7. Log v2 to MLflow and register it.
    with mlflow.start_run(run_name="V2_Retrained") as run:
        mlflow.log_param("data_sha256", calculate_data_hash())
        mlflow.log_param("training_rows", len(combined))
        mlflow.log_param("later_training_rows", len(first_70))
        mlflow.log_param("check_rows", len(last_30))
        mlflow.log_param("n_estimators", 300)
        mlflow.log_param("max_depth", "None")

        mlflow.log_metric("mae_v1_on_check", float(mae_v1))
        mlflow.log_metric("mae_v2_on_check", float(mae_v2))

        model_info = mlflow.sklearn.log_model(
            sk_model=v2,
            name="model",
            registered_model_name=MODEL_NAME,
            serialization_format="cloudpickle",
        )

        v2_version = model_info.registered_model_version

    # 8. Apply the exact 8% promotion rule.
    promote = mae_v2 <= 0.92 * mae_v1

    if promote:
        client.set_registered_model_alias(
            MODEL_NAME,
            "champion",
            str(v2_version),
        )
        decision = "promote"
    else:
        decision = "keep"

    print("\n=== V1 VS V2 CHECK ===")
    print(f"V1 MAE: {mae_v1:.4f}")
    print(f"V2 MAE: {mae_v2:.4f}")
    print(f"Required V2 MAE: {0.92 * mae_v1:.4f}")
    print("Decision:", decision)
    print("V2 registered version:", v2_version)

    # 9. Save results for findings.yaml.
    results = {
        "psi": psi_results,
        "most_drifted_feature": most_drifted,
        "mae_by_group": group_mae,
        "worst_served_group": worst_group,
        "mae_v1_on_check": float(mae_v1),
        "mae_v2_on_check": float(mae_v2),
        "decision": decision,
        "v2_version": str(v2_version),
        "v2_run_id": run.info.run_id,
    }

    output_path = ROOT_DIR / "reports" / "monitoring_results.json"
    output_path.write_text(
        pd.Series(results).to_json(indent=2),
        encoding="utf-8",
    )

    print("\nResults saved to:", output_path)


if __name__ == "__main__":
    main()
