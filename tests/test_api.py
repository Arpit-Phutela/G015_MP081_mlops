
import pandas as pd
import pytest
from fastapi.testclient import TestClient

import mp081.app as app_module
from mp081.cleaning import clean_data


client = TestClient(app_module.app)


class DummyModel:
    def predict(self, data):
        return [12.5]


def test_health_endpoint():
    response = client.get("/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


def test_valid_prediction(monkeypatch):
    monkeypatch.setattr(
        app_module,
        "get_model",
        lambda: DummyModel(),
    )

    payload = {
        "battery_start_pct": 40,
        "session_minutes": 120,
        "charger_kw": 30,
        "temp_c": 25,
        "vehicle_age_yr": 3,
        "queue_before": 2,
        "vehicle": "car",
        "station_type": "Mall",
    }

    response = client.post("/predict", json=payload)

    assert response.status_code == 200
    assert response.json()["prediction"] == 12.5
    assert response.json()["unit"] == "kWh"


@pytest.mark.parametrize(
    "field,value",
    [
        ("charger_kw", 151),
        ("temp_c", 2.4),
    ],
)
def test_invalid_prediction_inputs(field, value):
    payload = {
        "battery_start_pct": 40,
        "session_minutes": 120,
        "charger_kw": 30,
        "temp_c": 25,
        "vehicle_age_yr": 3,
        "queue_before": 2,
        "vehicle": "car",
        "station_type": "Mall",
    }

    payload[field] = value

    response = client.post("/predict", json=payload)

    assert response.status_code == 422


def test_clean_data():
    sample = pd.DataFrame({
        "date": ["2026-05-01", "2026-05-02", "2026-05-03"],
        "battery_start_pct": [40, 50, 60],
        "session_minutes": [999, 120, 140],
        "charger_kw": [30, 30, 30],
        "temp_c": [25, 26, 27],
        "vehicle_age_yr": [3, 4, 5],
        "queue_before": [1, 2, 3],
        "vehicle": [" CAR ", "car", "car"],
        "station_type": ["Mall", "Mall", "Mall"],
        "kwh_billed_in_app": [10, 20, 30],
        "energy_kwh": [100, 20, 30],
    })

    cleaned = clean_data(sample)

    assert "kwh_billed_in_app" not in cleaned.columns
    assert cleaned.loc[0, "energy_kwh"] == 10
    assert cleaned.loc[0, "session_minutes"] != 999
    assert cleaned.loc[0, "vehicle"] == "car"
