import argparse
import mlflow
import mlflow.sklearn
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

parser = argparse.ArgumentParser()
parser.add_argument("--n_estimators", type=int, default=100)
parser.add_argument("--max_depth", type=int, default=5)
args = parser.parse_args()

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

with mlflow.start_run():
    mlflow.log_param("n_estimators", args.n_estimators)
    mlflow.log_param("max_depth", args.max_depth)

    model = RandomForestClassifier(
        n_estimators=args.n_estimators, max_depth=args.max_depth, random_state=42
    )
    model.fit(X_train, y_train)

    acc = accuracy_score(y_test, model.predict(X_test))
    mlflow.log_metric("accuracy", acc)
    mlflow.sklearn.log_model(model, name="model", serialization_format="cloudpickle")

    print(f"n_estimators={args.n_estimators}, max_depth={args.max_depth}, accuracy={acc:.3f}")