# G015_MP081_mlops

# MP081 – EV Charging Session Energy Prediction

## Group Details

- **Group ID:** G015
- **Problem ID:** MP081
- **Project:** EV Charging Session Energy Prediction (Nagpur)

### Group Members

| Enrolment Number | GitHub / Name |
|---|---|
| S24CSEU1608 | Akshatgaba |
| S24CSEU1609 | SAMRIDDHISINGH |
| S24CSEU1594 | ArpitPhutela |

## Problem Statement

Build a machine learning regression system to predict the energy consumed (in kWh) during an EV charging session using session and vehicle features. The model is evaluated using Mean Absolute Error (MAE).

## Technology Stack

- Python 3.12
- Pandas and NumPy
- Scikit-learn
- MLflow
- FastAPI and Uvicorn
- Docker
- Kubernetes YAML manifests
- Pytest
- GitHub Actions
- DVC

## Project Structure

```text
.
├── .github/workflows/ci.yml
├── data/
│   └── raw/
├── docker_model/
├── k8s/
│   ├── deployment.yaml
│   └── service.yaml
├── predictions/
│   └── holdout_predictions.csv
├── reports/
│   ├── findings.yaml
│   └── monitoring_results.json
├── src/mp081/
│   ├── app.py
│   ├── cleaning.py
│   ├── data_quality.py
│   ├── monitoring.py
│   ├── predict.py
│   └── train.py
├── tests/
│   └── test_api.py
├── Dockerfile
└── requirements.txt
```

## Machine Learning Pipeline

1. Detect data quality issues in the supplied datasets.
2. Clean the data without modifying the original CSV files.
3. Train and evaluate Random Forest regression models.
4. Track experiments, metrics and model versions using MLflow.
5. Monitor feature drift using Population Stability Index (PSI).
6. Compare model performance across station types.
7. Evaluate a retrained model and promote it when the improvement meets the required threshold.
8. Generate predictions for the holdout dataset.

## Model Monitoring

- PSI is calculated for all six numerical input features.
- MAE is evaluated separately for Mall, Highway and Residential stations.
- The retrained model is promoted only when its MAE is at least 8% lower than the previous model's MAE on the comparison dataset.

## Run Locally

Install dependencies:

```bash
pip install -r requirements.txt
```

Set the Python module path:

```powershell
$env:PYTHONPATH = "$PWD\src"
```

Run the data quality checks:

```bash
python -m mp081.data_quality
```

Train the model:

```bash
python -m mp081.train
```

Run monitoring:

```bash
python -m mp081.monitoring
```

Generate holdout predictions:

```bash
python -m mp081.predict
```

Start the API:

```bash
uvicorn mp081.app:app --host 127.0.0.1 --port 8000
```

API health endpoint: `GET /health`

Prediction endpoint: `POST /predict`

## Run Tests

```bash
python -m pytest tests/ -v
```

## Docker

Build the image:

```bash
docker build -t mp081-api:1.0 .
```

Run the container:

```bash
docker run -d --name mp081-container -p 8000:8000 mp081-api:1.0
```

## Kubernetes

Kubernetes manifests are available in the `k8s/` directory:

- `deployment.yaml` defines two API replicas and health probes.
- `service.yaml` exposes the application on port 80 and forwards traffic to port 8000.

## Continuous Integration

GitHub Actions runs the automated test suite on pushes and pull requests to the `main` branch.

## Output

The final holdout predictions are stored in:

`predictions/holdout_predictions.csv`

The file contains exactly two columns: `record_id` and `prediction`.
