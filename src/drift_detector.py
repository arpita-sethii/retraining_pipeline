import numpy as np
from scipy.stats import wasserstein_distance, ks_2samp
from scipy.special import kl_div
import mlflow

def compute_psi(expected, actual, buckets=10):
    """Population Stability Index — detects input distribution shift."""
    def scale_range(x, min_val, max_val):
        return (x - min_val) / (max_val - min_val + 1e-8)

    min_val = min(expected.min(), actual.min())
    max_val = max(expected.max(), actual.max())

    expected_scaled = scale_range(expected, min_val, max_val)
    actual_scaled = scale_range(actual, min_val, max_val)

    breakpoints = np.linspace(0, 1, buckets + 1)

    expected_counts = np.histogram(expected_scaled, breakpoints)[0] + 1e-8
    actual_counts = np.histogram(actual_scaled, breakpoints)[0] + 1e-8

    expected_pct = expected_counts / expected_counts.sum()
    actual_pct = actual_counts / actual_counts.sum()

    psi = np.sum((actual_pct - expected_pct) * np.log(actual_pct / expected_pct))
    return psi

def compute_wasserstein(reference, current):
    """Wasserstein distance — how much distribution has shifted."""
    return wasserstein_distance(reference.flatten(), current.flatten())

def compute_kl_divergence(reference, current, buckets=10):
    """KL Divergence — asymmetric distribution difference."""
    min_val = min(reference.min(), current.min())
    max_val = max(reference.max(), current.max())
    breakpoints = np.linspace(min_val, max_val, buckets + 1)

    ref_counts = np.histogram(reference, breakpoints)[0] + 1e-8
    cur_counts = np.histogram(current, breakpoints)[0] + 1e-8

    ref_pct = ref_counts / ref_counts.sum()
    cur_pct = cur_counts / cur_counts.sum()

    return np.sum(kl_div(ref_pct, cur_pct))

def compute_ks_test(reference, current):
    """KS Test — statistical test for distribution difference."""
    stat, p_value = ks_2samp(reference.flatten(), current.flatten())
    return stat, p_value

def run_drift_report(reference_X, current_X, run_name="drift_report"):
    """
    Run full drift detection suite and log to MLflow.
    reference_X: training data (baseline distribution)
    current_X:   new incoming data chunk
    """
    ref = reference_X.flatten()
    cur = current_X.flatten()

    psi = compute_psi(ref, cur)
    wasserstein = compute_wasserstein(ref, cur)
    kl = compute_kl_divergence(ref, cur)
    ks_stat, ks_pvalue = compute_ks_test(ref, cur)

    # PSI interpretation
    if psi < 0.1:
        psi_status = "✅ No drift"
    elif psi < 0.2:
        psi_status = "⚠️  Moderate drift"
    else:
        psi_status = "🚨 Severe drift"

    # KS test interpretation
    ks_status = "🚨 Distributions differ" if ks_pvalue < 0.05 else "✅ Distributions similar"

    print(f"\n{'='*55}")
    print(f"📊  DRIFT DETECTION REPORT — {run_name}")
    print(f"{'='*55}")
    print(f"PSI Score        : {psi:.4f}  → {psi_status}")
    print(f"Wasserstein Dist : {wasserstein:.4f}")
    print(f"KL Divergence    : {kl:.4f}")
    print(f"KS Statistic     : {ks_stat:.4f}  → {ks_status}")
    print(f"KS p-value       : {ks_pvalue:.4f}")
    print(f"{'='*55}")

    # Log to MLflow
    with mlflow.start_run(run_name=run_name):
        mlflow.log_metrics({
            "psi_score": psi,
            "wasserstein_distance": wasserstein,
            "kl_divergence": kl,
            "ks_statistic": ks_stat,
            "ks_pvalue": ks_pvalue
        })
        mlflow.log_params({
            "psi_threshold_moderate": 0.1,
            "psi_threshold_severe": 0.2,
            "ks_significance_level": 0.05
        })

    return {
        "psi": psi,
        "wasserstein": wasserstein,
        "kl": kl,
        "ks_stat": ks_stat,
        "ks_pvalue": ks_pvalue,
        "data_drift_detected": psi > 0.1 or ks_pvalue < 0.05
    }