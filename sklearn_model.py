import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-random-forest")

# Data
X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Hyperparameters
params = {"n_estimators": 100, "max_depth": 5, "random_state": 42}

with mlflow.start_run(run_name="rf-baseline"):
    # Params log karo
    mlflow.log_params(params)

    # Train
    model = RandomForestClassifier(**params)
    model.fit(X_train, y_train)

    # Evaluate
    preds = model.predict(X_test)
    acc = accuracy_score(y_test, preds)
    f1 = f1_score(y_test, preds, average="macro")

    # Metrics log karo
    mlflow.log_metric("accuracy", acc)
    mlflow.log_metric("f1_macro", f1)

    # Model log karo (cloudpickle format, skops error se bachne ke liye)
    mlflow.sklearn.log_model(
        model,
        name="model",
        serialization_format="cloudpickle",
    )

    print(f"Accuracy: {acc:.3f}, F1: {f1:.3f}")