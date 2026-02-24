import torch
import torch.nn as nn
import mlflow
import mlflow.pytorch
import numpy as np
from torch.utils.data import DataLoader, TensorDataset
from src.model import LSTMForecaster
from mlflow import MlflowClient

def train_model(X_train, y_train, epochs=20, batch_size=64, lr=0.001, run_name="baseline"):

    X = torch.tensor(X_train, dtype=torch.float32)
    y = torch.tensor(y_train, dtype=torch.float32)

    dataset = TensorDataset(X, y)
    loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    model = LSTMForecaster()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.MSELoss()

    with mlflow.start_run(run_name=run_name) as run:

        mlflow.log_params({
            "epochs": epochs,
            "batch_size": batch_size,
            "learning_rate": lr,
            "hidden_size": 64,
            "num_layers": 2,
            "seq_len": X_train.shape[1]
        })

        for epoch in range(epochs):
            model.train()
            epoch_loss = 0

            for X_batch, y_batch in loader:
                optimizer.zero_grad()
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                loss.backward()
                optimizer.step()
                epoch_loss += loss.item()

            avg_loss = epoch_loss / len(loader)
            mlflow.log_metric("train_loss", avg_loss, step=epoch)

            if (epoch + 1) % 5 == 0:
                print(f"Epoch {epoch+1}/{epochs} — Loss: {avg_loss:.6f}")

        # Log model
        mlflow.pytorch.log_model(model, "model")

        # Register model in MLflow Model Registry
        model_uri = f"runs:/{run.info.run_id}/model"
        client = MlflowClient()

        try:
            client.create_registered_model("LSTMForecaster")
        except Exception:
            pass  # already exists

        mv = client.create_model_version(
            name="LSTMForecaster",
            source=model_uri,
            run_id=run.info.run_id
        )
        print(f"Model registered as LSTMForecaster version {mv.version}")
        print(f"Run '{run_name}' complete. Model logged to MLflow.")

    return model