import mlflow

# Tracking server se connect karo (jo UI chal rahi hai)
mlflow.set_tracking_uri("http://127.0.0.1:5000")

# Experiment ka naam set karo (nahi hoga to ban jayega)
mlflow.set_experiment("my-first-experiment")

with mlflow.start_run(run_name="run-1"):
    # Parameters: model ki settings
    mlflow.log_param("learning_rate", 0.01)
    mlflow.log_param("epochs", 10)

    # Metrics: results
    mlflow.log_metric("accuracy", 0.85)
    mlflow.log_metric("loss", 0.35)

    print("Run complete!")