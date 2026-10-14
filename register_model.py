import mlflow
from mlflow.tracking import MlflowClient

mlflow.set_tracking_uri("http://127.0.0.1:5000")
client = MlflowClient()

# Sahi run (jisme model save hua tha)
run_id = "5175bcefb2734a249eac1f298ffcde13"
model_name = "iris-classifier"

# 1. Register karo: pehle purana tareeqa, fail ho to logged model se
try:
    result = mlflow.register_model(f"runs:/{run_id}/model", model_name)
except Exception as e:
    print("runs:/ fail hua, logged model se try kar raha hoon:", e)
    exp = client.get_experiment_by_name("iris-random-forest")
    logged = mlflow.search_logged_models(
        experiment_ids=[exp.experiment_id], output_format="list"
    )
    mine = [m for m in logged if m.source_run_id == run_id][0]
    result = mlflow.register_model(mine.model_uri, model_name)

print("Registered:", result.name, "version", result.version)

# 2. Alias lagao
client.set_registered_model_alias(model_name, "champion", result.version)

# 3. Description do
client.update_registered_model(model_name, description="Iris RandomForest classifier")
print("Alias 'champion' set ho gaya")