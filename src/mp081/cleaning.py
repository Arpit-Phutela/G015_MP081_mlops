
from pathlib import Path
import pandas as pd

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"

# Features available when a charging session begins.
NUMERIC_FEATURES = [
    "battery_start_pct",
    "session_minutes",
    "charger_kw",
    "temp_c",
    "vehicle_age_yr",
    "queue_before",
]

CATEGORICAL_FEATURES = [
    "vehicle",
    "station_type",
]

FEATURES = NUMERIC_FEATURES + CATEGORICAL_FEATURES


def clean_data(df):
    """
    Clean a copy of the supplied dataset.
    The original DataFrame and CSV files remain unchanged.
    """
    data = df.copy()

    # 1. Remove duplicate records.
    data = data.drop_duplicates().copy()

    # 2. Remove the post-session leakage column.
    if "kwh_billed_in_app" in data.columns:
        data = data.drop(columns=["kwh_billed_in_app"])

    # 3. Replace impossible sentinel session durations.
    data.loc[data["session_minutes"] == 999, "session_minutes"] = pd.NA

    # 4. Treat the stuck queue sensor period as unreliable.
    if "date" in data.columns:
        dates = pd.to_datetime(data["date"], errors="coerce")
        stuck_period = dates.between(
            pd.Timestamp("2026-04-07"),
            pd.Timestamp("2026-04-30"),
        )
        data.loc[stuck_period, "queue_before"] = pd.NA

    # 5. Correct target values typed ten times too large.
    if "energy_kwh" in data.columns and "kwh_billed_in_app" in df.columns:
        billed = df.loc[data.index, "kwh_billed_in_app"]
        ratio = data["energy_kwh"] / billed
        typo_rows = ratio.between(9, 11)
        data.loc[typo_rows, "energy_kwh"] = (
            data.loc[typo_rows, "energy_kwh"] / 10
        )

    # 6. Normalize text categories.
    for col in CATEGORICAL_FEATURES:
        data[col] = data[col].astype("string").str.strip().str.lower()
        data[col] = data[col].replace({
            "two wheeler": "two_wheeler",
            "three wheeler": "three_wheeler",
        })

    # 7. Impute missing numeric feature values using medians.
    for col in NUMERIC_FEATURES:
        data[col] = pd.to_numeric(data[col], errors="coerce")
        data[col] = data[col].fillna(data[col].median())

    return data


def load_clean_data():
    """Load and clean training and later datasets."""
    train = pd.read_csv(DATA_DIR / "train.csv")
    later = pd.read_csv(DATA_DIR / "later.csv")

    clean_train = clean_data(train)
    clean_later = clean_data(later)

    return clean_train, clean_later


if __name__ == "__main__":
    train, later = load_clean_data()

    print("=== CLEANING SUMMARY ===")
    print(f"Clean training rows: {len(train)}")
    print(f"Clean later rows: {len(later)}")
    print(f"Training missing values:\n{train.isnull().sum()}")
    print(f"Later missing values:\n{later.isnull().sum()}")

    print("\nTraining feature columns:")
    print(FEATURES)
