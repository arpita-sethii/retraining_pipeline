# 🔁 Automated ML Retraining Pipeline
### LSTM Time Series Forecasting · Drift Detection · MLflow · FastAPI

> *"Most ML projects end at model training. This one starts there."*

---

## What this actually does

This pipeline trains an LSTM to forecast electricity transformer oil temperature, then watches its own performance over time — and retrains itself automatically when it detects the data has drifted.

No manual intervention. No threshold-tweaking by hand. Just a system that monitors itself, catches degradation early, and recovers.

Built on the **ETTh1 dataset** — the same benchmark used in Informer, PatchTST, and other landmark time series papers.

---

## The problem it solves

In production, models don't fail dramatically. They degrade quietly. Data distributions shift, seasonal patterns change, and your model keeps predicting with false confidence while accuracy slowly erodes.

Most teams catch this weeks later — after it's already caused damage.

This pipeline catches it **before performance degrades**, using statistical drift tests on the input data itself.

---

## Architecture

```
ETTh1 Dataset (2016 → 2018)
        │
        ▼
┌───────────────────┐
│   Data Chunker    │  Splits into 3 chronological windows
│  chunk1 / 2 / 3  │  to simulate real temporal drift
└────────┬──────────┘
         │
         ▼
┌───────────────────┐
│  Baseline Train   │  2-layer LSTM, 64 hidden units
│  (Chunk 1 only)   │  Adam optimizer, MSE loss
└────────┬──────────┘
         │
         ▼
┌───────────────────────────────────┐
│       Drift Detection Suite       │
│  ┌─────────────────────────────┐  │
│  │ PSI Score (distribution)    │  │
│  │ Wasserstein Distance        │  │
│  │ KL Divergence               │  │
│  │ KS Test (statistical)       │  │
│  └─────────────────────────────┘  │
└────────┬──────────────────────────┘
         │
         ▼
┌───────────────────┐     drift?      ┌──────────────────┐
│  MSE Evaluation   │ ─── YES ──────► │   Auto Retrain   │
│  vs threshold     │                 │  on expanded data│
└────────┬──────────┘                 └────────┬─────────┘
         │                                     │
         └─────────────┬───────────────────────┘
                       ▼
            ┌─────────────────────┐
            │  MLflow Experiment  │
            │  Tracking + Model   │
            │  Registry (versioned│
            └─────────────────────┘
                       │
                       ▼
            ┌─────────────────────┐
            │   FastAPI Endpoint  │
            │  /predict /health   │
            │  /reload-model      │
            └─────────────────────┘
```

---

## Drift Detection — the interesting part

Most pipelines detect drift by watching MSE. That's reactive — you only know the model is broken after it's already broken.

This pipeline runs **4 statistical tests on the input data itself**, before evaluating model performance:

| Metric | What it measures | Threshold |
|--------|-----------------|-----------|
| **PSI Score** | Distribution shift magnitude | > 0.1 moderate, > 0.2 severe |
| **Wasserstein Distance** | How far distributions have moved | logged for trend |
| **KL Divergence** | Asymmetric information difference | logged for trend |
| **KS Test** | Statistical significance of shift | p-value < 0.05 |

**Real results on ETTh1:**

```
Chunk 1 vs Chunk 2:
  PSI Score        : 4.34  → 🚨 Severe drift
  Wasserstein Dist : 0.08
  KL Divergence    : 3.94
  KS Statistic     : 0.23  → 🚨 Distributions differ (p=0.000)

Chunk 1 vs Chunk 3:
  PSI Score        : 9.70  → 🚨 Severe drift (2.2x worse)
  Wasserstein Dist : 0.21
  KL Divergence    : 8.91
  KS Statistic     : 0.54  → 🚨 Distributions differ (p=0.000)
```

This is real seasonal drift in the ETTh1 data — not simulated, not artificially injected.

---

## Results

After automated retraining triggered on Chunk 2 drift:

```
┌─────────────────────────┬──────────┬──────────┬────────┐
│ Run                     │   MSE    │   MAE    │ Drift  │
├─────────────────────────┼──────────┼──────────┼────────┤
│ chunk2_evaluation       │ 0.000473 │ 0.015642 │ ⚠️ YES │
│ chunk3_evaluation       │ 0.000160 │ 0.008913 │ ✅ NO  │
├─────────────────────────┼──────────┼──────────┼────────┤
│ Overall improvement     │  +66.3%  │  +43.0%  │        │
└─────────────────────────┴──────────┴──────────┴────────┘
```

MSE dropped 66.3% after retraining. The model recovered.

---

## Project Structure

```
retraining_pipeline/
├── src/
│   ├── data_loader.py      # ETTh1 loading, scaling, chunking, sequence creation
│   ├── model.py            # 2-layer LSTM architecture
│   ├── train.py            # Training loop + MLflow logging + Model Registry
│   ├── evaluate.py         # MSE/MAE evaluation + drift trigger + comparison table
│   ├── drift_detector.py   # PSI, Wasserstein, KL divergence, KS test
│   ├── pipeline.py         # Full orchestration + MSE trend plot
│   ├── scheduler.py        # APScheduler automated runs
│   └── app.py              # FastAPI inference endpoint
├── ETTh1.csv
├── requirements.txt
└── model.pt                # Saved model weights
```

---

## Setup

```bash
git clone https://github.com/arpita-sethii/retraining-pipeline.git
cd retraining-pipeline

python -m venv venv
venv\Scripts\activate        # Windows
source venv/bin/activate     # Mac/Linux

pip install -r requirements.txt
```

---

## Run

**Train + evaluate + detect drift (single run):**
```bash
python -m src.pipeline
```

**Automated scheduler (runs every 30 seconds):**
```bash
python -m src.scheduler
```

**MLflow UI (track experiments):**
```bash
mlflow ui
# open http://127.0.0.1:5000
```

**FastAPI inference endpoint:**
```bash
uvicorn src.app:app --reload --port 8000
# open http://127.0.0.1:8000/docs
```

---

## API Usage

**Health check:**
```bash
GET http://127.0.0.1:8000/health
```

**Predict next hour temperature:**
```bash
POST http://127.0.0.1:8000/predict
{
  "sequence": [0.5, 0.48, 0.51, ...]  # 24 scaled values
}
```

**Reload latest model from registry:**
```bash
POST http://127.0.0.1:8000/reload-model
```

---

## Dataset

**ETTh1 (Electricity Transformer Temperature)**
- 17,420 hourly readings from 2016–2018
- 7 features: HUFL, HULL, MUFL, MULL, LUFL, LULL, OT (target)
- Same benchmark used in Informer (AAAI 2021) and PatchTST papers
- Source: [zhouhaoyi/ETDataset](https://github.com/zhouhaoyi/ETDataset)

---

## Tech Stack

![Python](https://img.shields.io/badge/Python-3.11-0d0d0d?style=flat-square&logo=python&logoColor=9B59F7)
![PyTorch](https://img.shields.io/badge/PyTorch-2.3-0d0d0d?style=flat-square&logo=pytorch&logoColor=9B59F7)
![MLflow](https://img.shields.io/badge/MLflow-2.19-0d0d0d?style=flat-square&logo=mlflow&logoColor=9B59F7)
![FastAPI](https://img.shields.io/badge/FastAPI-0d0d0d?style=flat-square&logo=fastapi&logoColor=9B59F7)
![SciPy](https://img.shields.io/badge/SciPy-0d0d0d?style=flat-square&logo=scipy&logoColor=9B59F7)

---

## What's next

- [ ] Replace LSTM with Informer/PatchTST and benchmark
- [ ] Event-driven retraining on data arrival instead of scheduler
- [ ] Deploy FastAPI endpoint to Render
- [ ] Add feature drift detection per column (not just target)

---

*Built by [Arpita Sethi](https://github.com/arpita-sethii) · [Portfolio](https://arpita-sethi-portfolio-website.vercel.app/) · [LinkedIn](https://linkedin.com/in/arpita-sethi)*
