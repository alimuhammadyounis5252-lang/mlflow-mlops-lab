import mlflow
import mlflow.sklearn
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from mlflow.tracking import MlflowClient
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, ConfusionMatrixDisplay

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-pipeline")
client = MlflowClient()
MODEL_NAME = "iris-pipeline-model"

# 1. Data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.3, random_state=7
)

# 2. Candidate models
candidates = [
    ("rf-small", RandomForestClassifier(n_estimators=5, max_depth=2, random_state=42)),
    ("rf-big", RandomForestClassifier(n_estimators=200, max_depth=8, random_state=42)),
    ("logreg", LogisticRegression(max_iter=300)),
]

# 3. Train aur track (parent run ke andar child runs)
with mlflow.start_run(run_name="pipeline") as parent:
    for name, model in candidates:
        with mlflow.start_run(run_name=name, nested=True):
            mlflow.log_param("model_type", type(model).__name__)
            mlflow.log_params(model.get_params())

            model.fit(X_train, y_train)
            preds = model.predict(X_test)
            mlflow.log_metric("accuracy", accuracy_score(y_test, preds))
            mlflow.log_metric("f1_macro", f1_score(y_test, preds, average="macro"))

            fig, ax = plt.subplots()
            ConfusionMatrixDisplay.from_predictions(y_test, preds, ax=ax)
            mlflow.log_figure(fig, "confusion_matrix.png")
            plt.close(fig)

            info = mlflow.sklearn.log_model(
                model, name="model", serialization_format="cloudpickle"
            )
            mlflow.set_tag("model_uri", info.model_uri)
            print(f"{name}: done")

    # 4. Best model chuno
    best = mlflow.search_runs(
        filter_string=f"tags.mlflow.parentRunId = '{parent.info.run_id}'",
        order_by=["metrics.accuracy DESC", "metrics.f1_macro DESC"],
        max_results=1,
    ).iloc[0]
    print("\nBest:", best["tags.mlflow.runName"], "accuracy =", best["metrics.accuracy"])

    # 5. Registry mein register + alias
    mv = mlflow.register_model(best["tags.model_uri"], MODEL_NAME)
    client.set_registered_model_alias(MODEL_NAME, "champion", mv.version)
    print(f"Registered {MODEL_NAME} v{mv.version} as champion")

# 6. Alias se load karke predict
loaded = mlflow.pyfunc.load_model(f"models:/{MODEL_NAME}@champion")
print("Predictions:", loaded.predict(X_test[:5]), "| Actual:", y_test[:5])