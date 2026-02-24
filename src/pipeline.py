import mlflow
import mlflow.pytorch
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from src.data_loader import get_chunks
from src.train import train_model
from src.evaluate import evaluate_model, print_comparison_table, evaluation_history
from src.drift_detector import run_drift_report
import torch


DATA_PATH = "ETTh1.csv"
SEQ_LEN = 24
THRESHOLD = 0.0003

def plot_mse_trend(mse_values, labels, threshold, save_path="mse_trend.png"):
    plt.figure(figsize=(10, 5))
    plt.style.use("dark_background")

    colors = ["#9B59F7" if mse <= threshold else "#FF4C4C" for mse in mse_values]
    bars = plt.bar(labels, mse_values, color=colors, width=0.5, zorder=3)

    plt.axhline(y=threshold, color="#FF4C4C", linestyle="--",
                linewidth=2, label=f"Threshold = {threshold}", zorder=4)

    for bar, mse in zip(bars, mse_values):
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            bar.get_height() + 0.000005,
            f"{mse:.6f}",
            ha="center", va="bottom",
            color="white", fontsize=10, fontweight="bold"
        )

    plt.title("MSE Over Time — Drift Detection & Retraining Pipeline",
              fontsize=14, fontweight="bold", pad=15)
    plt.xlabel("Evaluation Stage", fontsize=12)
    plt.ylabel("MSE", fontsize=12)

    healthy = mpatches.Patch(color="#9B59F7", label="Healthy (below threshold)")
    drifted = mpatches.Patch(color="#FF4C4C", label="Drift detected (above threshold)")
    plt.legend(handles=[healthy, drifted,
               plt.Line2D([0], [0], color="#FF4C4C",
               linestyle="--", label=f"Threshold = {threshold}")],
               loc="upper right", fontsize=9)

    plt.tight_layout()
    plt.savefig(save_path, dpi=150, bbox_inches="tight")
    plt.close()
    print(f"\n📈 MSE trend plot saved to {save_path}")
    return save_path


def run_pipeline():
    mlflow.set_experiment("retraining_pipeline")
    evaluation_history.clear()

    print("=" * 55)
    print("STEP 1: Loading data chunks...")
    print("=" * 55)
    (X1, y1), (X2, y2), (X3, y3), scaler = get_chunks(DATA_PATH, SEQ_LEN)

    print("\n" + "=" * 55)
    print("STEP 2: Training baseline on Chunk 1 (2016 data)...")
    print("=" * 55)
    model = train_model(X1, y1, epochs=20, run_name="baseline")

    # Data drift: chunk1 vs chunk2
    print("\n" + "=" * 55)
    print("STEP 3: Data drift detection — Chunk 1 vs Chunk 2...")
    print("=" * 55)
    drift_report_2 = run_drift_report(X1, X2, run_name="drift_chunk1_vs_chunk2")

    print("\n" + "=" * 55)
    print("STEP 4: Performance evaluation on Chunk 2...")
    print("=" * 55)
    mse2, needs_retraining = evaluate_model(
        model, X2, y2,
        run_name="chunk2_evaluation",
        threshold=THRESHOLD
    )

    # Summarize drift situation
    print(f"\n🔍 Drift Summary (Chunk 2):")
    print(f"   Data drift    : {'YES' if drift_report_2['data_drift_detected'] else 'NO'}")
    print(f"   Perf drift    : {'YES' if needs_retraining else 'NO'}")

    if needs_retraining or drift_report_2["data_drift_detected"]:
        print("\n" + "=" * 55)
        print("STEP 5: Retraining triggered — Chunk 1 + 2...")
        print("=" * 55)
        X_retrain = np.concatenate([X1, X2], axis=0)
        y_retrain = np.concatenate([y1, y2], axis=0)
        model = train_model(
            X_retrain, y_retrain,
            epochs=30,
            lr=0.0005,
            run_name="retrained"
        )

    # Data drift: chunk1 vs chunk3
    print("\n" + "=" * 55)
    print("STEP 6: Data drift detection — Chunk 1 vs Chunk 3...")
    print("=" * 55)
    drift_report_3 = run_drift_report(X1, X3, run_name="drift_chunk1_vs_chunk3")

    print("\n" + "=" * 55)
    print("STEP 7: Performance evaluation on Chunk 3...")
    print("=" * 55)
    mse3, needs_retraining2 = evaluate_model(
        model, X3, y3,
        run_name="chunk3_evaluation",
        threshold=THRESHOLD
    )

    print(f"\n🔍 Drift Summary (Chunk 3):")
    print(f"   Data drift    : {'YES' if drift_report_3['data_drift_detected'] else 'NO'}")
    print(f"   Perf drift    : {'YES' if needs_retraining2 else 'NO'}")

    if needs_retraining2 or drift_report_3["data_drift_detected"]:
        print("\n" + "=" * 55)
        print("STEP 8: Final retrain on all data...")
        print("=" * 55)
        X_all = np.concatenate([X1, X2, X3], axis=0)
        y_all = np.concatenate([y1, y2, y3], axis=0)
        model = train_model(
            X_all, y_all,
            epochs=30,
            lr=0.0003,
            run_name="final_retrained"
        )
        evaluate_model(
            model, X3, y3,
            run_name="final_evaluation",
            threshold=THRESHOLD
        )

    # MSE trend plot
    mse_vals = [e["mse"] for e in evaluation_history]
    labels = [e["run"].replace("_", "\n") for e in evaluation_history]
    plot_path = plot_mse_trend(mse_vals, labels, THRESHOLD)

    with mlflow.start_run(run_name="mse_trend_plot"):
        mlflow.log_artifact(plot_path)

    # Comparison table
    print_comparison_table()
    # Save final model to file for deployment
    torch.save(model.state_dict(), "model.pt")
    print("💾 Model saved to model.pt")

    print("\n" + "=" * 55)
    print("✅ Pipeline complete.")
    print("=" * 55)


if __name__ == "__main__":
    run_pipeline()