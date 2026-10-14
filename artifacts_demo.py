import mlflow
import pandas as pd
import matplotlib
matplotlib.use("Agg")  # GUI window nahi khulegi, seedha file save hogi
import matplotlib.pyplot as plt

mlflow.set_tracking_uri("http://127.0.0.1:5000")
mlflow.set_experiment("artifacts-demo")

with mlflow.start_run(run_name="artifacts-run"):
    mlflow.log_param("model", "demo")
    mlflow.log_metric("accuracy", 0.88)

    # 1. Text file
    with open("notes.txt", "w") as f:
        f.write("Ye mera pehla artifact hai.\nLearning rate 0.01 use kiya.")
    mlflow.log_artifact("notes.txt")

    # 2. Dataset (CSV)
    df = pd.DataFrame({
        "epoch": [1, 2, 3, 4, 5],
        "loss": [0.9, 0.6, 0.45, 0.35, 0.30],
    })
    df.to_csv("loss_data.csv", index=False)
    mlflow.log_artifact("loss_data.csv")

    # 3. Plot
    plt.plot(df["epoch"], df["loss"], marker="o")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.savefig("loss_plot.png")
    plt.close()
    mlflow.log_artifact("loss_plot.png")

    # 4. Folder mein organize karke
    mlflow.log_artifact("loss_plot.png", artifact_path="plots")

    print("Artifacts logged!")