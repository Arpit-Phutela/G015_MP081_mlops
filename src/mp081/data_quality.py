
from pathlib import Path
import pandas as pd
from difflib import SequenceMatcher

ROOT_DIR = Path(__file__).resolve().parents[2]
DATA_DIR = ROOT_DIR / "data" / "raw"
train = pd.read_csv(DATA_DIR / "train.csv")
later = pd.read_csv(DATA_DIR / "later.csv")
holdout = pd.read_csv(DATA_DIR / "holdout_features.csv")


def detect_leakage():
    """Find columns present in labelled data but absent in holdout."""
    return [
        col for col in train.columns
        if col not in holdout.columns and col not in ["energy_kwh", "record_id", "date"]
    ]


def detect_sentinel():
    """Find numeric columns containing the suspicious sentinel 999."""
    results = {}

    for col in train.select_dtypes(include="number").columns:
        count = int((train[col] == 999).sum())
        if count:
            results[col] = count

    return results


def detect_duplicates():
    """Count exact duplicate training rows."""
    return int(train.duplicated().sum())


def detect_target_typos():
    """Find target values approximately ten times the billed value."""
    ratio = train["energy_kwh"] / train["kwh_billed_in_app"]
    return int(ratio.between(9, 11).sum())


def detect_stuck_sensor():
    """Find long consecutive date ranges with one constant daily value."""
    data = train.copy()
    data["date"] = pd.to_datetime(data["date"])

    candidates = []

    for col in ["battery_start_pct", "session_minutes",
                "charger_kw", "temp_c", "vehicle_age_yr", "queue_before"]:

        daily = data.groupby("date")[col].agg(["nunique", "first"])
        constant_days = daily[daily["nunique"] == 1]

        for value, group in constant_days.groupby("first"):
            dates = group.index.sort_values()

            if len(dates) < 7:
                continue

            start = previous = dates[0]

            for current in dates[1:]:
                if (current - previous).days > 1:
                    if (previous - start).days >= 6:
                        candidates.append(
                            (col, value, start.date(), previous.date())
                        )
                    start = current
                previous = current

            if (previous - start).days >= 6:
                candidates.append(
                    (col, value, start.date(), previous.date())
                )

    return candidates


def inspect_spelling_variants():
    """Show category values that differ only by case or whitespace."""
    results = {}

    for col in ["vehicle", "station_type"]:
        values = train[col].dropna().unique()
        normalized = {}

        for value in values:
            key = str(value).strip().casefold()
            normalized.setdefault(key, []).append(value)

        variants = {
            key: original
            for key, original in normalized.items()
            if len(original) > 1
        }

        results[col] = {
            "original_values": sorted(map(str, values)),
            "normalization_variants": variants,
            "unique_levels_after_normalization": len(normalized),
        }

    return results


def inspect_unit_mixup():
    """Compare feature medians across station groups and time periods."""
    columns = [
        "battery_start_pct", "session_minutes", "charger_kw",
        "temp_c", "vehicle_age_yr", "queue_before"
    ]

    print("\nTRAIN MEDIANS BY STATION:")
    print(train.groupby("station_type")[columns].median().round(2))

    print("\nLATER MEDIANS BY STATION:")
    print(later.groupby("station_type")[columns].median().round(2))


if __name__ == "__main__":
    print("=== DATA QUALITY DETECTION ===")

    print("\n1. Possible leaking columns:")
    print(detect_leakage())

    print("\n2. Sentinel values (value 999):")
    print(detect_sentinel())

    print("\n3. Exact duplicate rows:")
    print(detect_duplicates())

    print("\n4. Possible target typos (10x):")
    print(detect_target_typos())

    print("\n5. Possible stuck sensors:")
    for result in detect_stuck_sensor():
        print(result)

    print("\n6. Category spelling inspection:")
    print(inspect_spelling_variants())

    print("\n7. Unit-mixup investigation:")
    inspect_unit_mixup()
