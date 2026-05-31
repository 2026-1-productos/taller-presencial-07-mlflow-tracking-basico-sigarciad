"""Prediccion script for the MLflow model.

This script loads a model from MLflow and makes predictions on a dataset.

$ python3 make_predictions.py

"""

import mlflow
import pandas as pd

FILE_PATH = "data/winequality-red.csv"
EXPERIMENT_NAME = "wine_quality_experiment"

df = pd.read_csv(FILE_PATH)
y = df["quality"]
x = df.drop(columns=["quality"])

## En lugar de fijar el run_id a mano, se busca automaticamente el
## ultimo run registrado en el experimento. Asi el script funciona en
## cualquier maquina sin tener que editar el run_id.
experiment = mlflow.get_experiment_by_name(EXPERIMENT_NAME)
if experiment is None:
    raise RuntimeError(
        f"No existe el experimento '{EXPERIMENT_NAME}'. "
        "Entrene primero un modelo, por ejemplo: "
        "python3 -m homework --model knn"
    )

runs = mlflow.search_runs(
    experiment_ids=[experiment.experiment_id],
    order_by=["start_time DESC"],
    max_results=1,
)
if runs.empty:
    raise RuntimeError(
        "No hay runs registrados en el experimento. "
        "Ejecute primero: python3 -m homework --model knn"
    )

run_id = runs.iloc[0]["run_id"]
logged_model = f"runs:/{run_id}/model"

loaded_model = mlflow.pyfunc.load_model(logged_model)
y_pred = loaded_model.predict(x)

print(y_pred)
