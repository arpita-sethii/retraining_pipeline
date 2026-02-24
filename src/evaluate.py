import torch
import torch.nn as nn
import mlflow
import numpy as np
from src.model import LSTMForecaster

# Store results globally for comparison table
evaluation_history = []

def evaluate_model(model, X_test, y_test, run_name="evaluation", threshold=0.002):

    model.eval()
    criterion = nn.MSELoss()

    X = torch.tensor(X_test, dtype=torch.float32)
    y = torch.tensor(y_test, dtype=torch.float32)

    with torch.no_grad():
        preds = model(X)
        mse = criterion(preds, y).item()
        mae = torch.mean(torch.abs(preds - y)).item()

    print(f"\n--- Evaluation: {run_name} ---")
    print(f"MSE  : {mse:.6f}")
    print(f"MAE  : {mae:.6f}")
    print(f"Threshold : {threshold}")

    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics({
            "test_mse": mse,
            "test_mae": mae,
            "threshold": threshold
        })

    needs_retraining = mse > threshold

    if needs_retraining:
        print(f"\n⚠️  MSE {mse:.6f} > threshold — RETRAINING TRIGGERED")
    else:
        print(f"\n✅  MSE {mse:.6f} <= threshold — Model is healthy")

    # Store for comparison table
    evaluation_history.append({
        "run": run_name,
        "mse": mse,
        "mae": mae,
        "threshold": threshold,
        "drift": needs_retraining
    })

    return mse, needs_retraining


def print_comparison_table():
    if not evaluation_history:
        return

    print("\n" + "=" * 65)
    print("📊  MODEL PERFORMANCE COMPARISON TABLE")
    print("=" * 65)
    print(f"{'Run':<25} {'MSE':>10} {'MAE':>10} {'Drift':>10}")
    print("-" * 65)

    baseline_mse = evaluation_history[0]["mse"]
    baseline_mae = evaluation_history[0]["mae"]

    for i, entry in enumerate(evaluation_history):
        drift_flag = "⚠️  YES" if entry["drift"] else "✅  NO"

        if i == 0:
            improvement_mse = "—"
            improvement_mae = "—"
        else:
            imp_mse = ((baseline_mse - entry["mse"]) / baseline_mse) * 100
            imp_mae = ((baseline_mae - entry["mae"]) / baseline_mae) * 100
            improvement_mse = f"{imp_mse:+.1f}%"
            improvement_mae = f"{imp_mae:+.1f}%"

        print(f"{entry['run']:<25} {entry['mse']:>10.6f} {entry['mae']:>10.6f} {drift_flag:>10}")

    print("-" * 65)

    if len(evaluation_history) > 1:
        last = evaluation_history[-1]
        mse_change = ((baseline_mse - last["mse"]) / baseline_mse) * 100
        mae_change = ((baseline_mae - last["mae"]) / baseline_mae) * 100
        print(f"{'Overall improvement':<25} {mse_change:>+9.1f}% {mae_change:>+9.1f}%")

    print("=" * 65)