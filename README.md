# MLflow MLOps Lab

Hands-on MLOps lab covering the full MLflow workflow, built with scikit-learn on the Iris dataset.

## Tasks covered
1. Install MLflow and explore the UI
2. First tracked experiment (params, metrics)
3. Logging artifacts (files, plots, datasets)
4. Training and logging a scikit-learn model
5. Autolog
6. Querying runs with MlflowClient
7. Model Registry (versions, aliases)
8. Loading and serving a registered model (REST API)
9. MLflow Projects (reproducible runs)
10. End-to-end pipeline (nested runs, best-model selection, registration)

## Setup
```
python -m venv venv
venv\Scripts\activate
pip install mlflow scikit-learn pandas matplotlib requests
```

## Run
Start the MLflow UI in one terminal:
```
mlflow ui
```

Then run any script in another terminal, for example:
```
python pipeline.py
```

To run the MLflow Project:
```
cd iris_project
mlflow run . --env-manager local --experiment-name iris-project
```

## Files
- `first_experiment.py` to `query_runs.py`: tracking, artifacts, autolog, querying
- `register_model.py`, `load_model.py`, `call_api.py`: registry, loading, REST serving
- `iris_project/`: MLflow Project (`MLproject`, `train.py`, `python_env.yaml`)
- `pipeline.py`: end-to-end pipeline

## Author
Ali Muhammad Younis