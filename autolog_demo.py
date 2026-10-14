import mlflow
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("iris-autolog")

# Ye ek line sab kuch automatically log karegi
mlflow.sklearn.autolog(serialization_format="cloudpickle")

X, y = load_iris(return_X_y=True)
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Run 1: Random Forest
with mlflow.start_run(run_name="autolog-rf"):
    model = RandomForestClassifier(n_estimators=50, max_depth=3, random_state=42)
    model.fit(X_train, y_train)
    print("RF score:", model.score(X_test, y_test))

# Run 2: Logistic Regression
with mlflow.start_run(run_name="autolog-lr"):
    model = LogisticRegression(max_iter=200)
    model.fit(X_train, y_train)
    print("LR score:", model.score(X_test, y_test))