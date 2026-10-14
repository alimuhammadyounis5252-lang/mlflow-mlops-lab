import mlflow
import numpy as np

mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Alias se load karo, version number yaad rakhne ki zaroorat nahi
model = mlflow.pyfunc.load_model("models:/iris-classifier@champion")

samples = np.array([
    [5.1, 3.5, 1.4, 0.2],   # setosa (0)
    [6.7, 3.0, 5.2, 2.3],   # virginica (2)
    [5.9, 3.0, 4.2, 1.5],   # versicolor (1)
])

print("Predictions:", model.predict(samples))