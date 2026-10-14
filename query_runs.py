import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# 1. Sab experiments ki list
print("=== Experiments ===")
for exp in client.search_experiments():
    print(exp.experiment_id, exp.name)

# 2. Ek experiment ke runs (DataFrame mein)
exp = client.get_experiment_by_name("iris-random-forest")
runs_df = mlflow.search_runs(experiment_ids=[exp.experiment_id])
print("\n=== Runs DataFrame columns ===")
print(runs_df.columns.tolist())

print("\n=== Runs summary ===")
print(runs_df[["run_id", "tags.mlflow.runName", "metrics.accuracy", "metrics.f1_macro"]])

# 3. Filter ke saath search
best = mlflow.search_runs(
    experiment_ids=[exp.experiment_id],
    filter_string="metrics.accuracy > 0.9",
    order_by=["metrics.accuracy DESC"],
    max_results=1,
)
print("\n=== Best run ===")
print(best[["run_id", "metrics.accuracy"]])

# 4. Ek run ki detail client se
run_id = best.iloc[0]["run_id"]
run = client.get_run(run_id)
print("\nParams:", run.data.params)
print("Metrics:", run.data.metrics)
print("Status:", run.info.status)

# 5. Run ke artifacts ki list
print("\nArtifacts:")
for a in client.list_artifacts(run_id):
    print(" -", a.path)