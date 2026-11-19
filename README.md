# 💳 Credit Card Fraud Detection — M.Tech Project

A complete, reproducible machine-learning pipeline that detects fraudulent credit card transactions.  
Training runs on **Kaggle** (free GPU) and inference runs on your **local laptop**.

---

## ⚡ Quick Start (model already trained)

**You do not need to retrain anything.** The trained model is already in `models/`.

```bash
pip install -r requirements.txt
streamlit run app.py
```

Then upload `sample_with_fraud.csv` to see it in action.

To confirm the shipped model reproduces the published numbers:

```bash
python validate_model.py --test-split
```

To retrain from scratch on this machine (CPU is fine, takes longer than Kaggle GPU):

```bash
python model_training.py
```

Step 1 below is only needed if you want to train from scratch on Kaggle.

---

## 📁 Project Structure

```
Credit Card/
├── creditcard.csv                        ← raw dataset (auto-downloaded)
├── notebook/
│   └── fraud_detection_training.ipynb   ← run this on Kaggle
│
│   ── Project modules (report sections 3.1–3.7) ──
├── model_training.py                     ← 3.1 training the ML model
├── preprocess.py                         ← 3.2 loading and preprocessing
├── app.py                                ← 3.3 Streamlit application + CSV upload
├── prediction.py                         ← 3.4 fraud probability + classification
├── evaluation.py                         ← 3.5 accuracy / precision / recall / F1 / CM
├── visualization.py                      ← 3.6 fraud statistics + distribution
├── download.py                           ← 3.7 CSV download
│
├── preprocessing.py                      ← backward-compatible alias for preprocess.py
├── predict.py                            ← command-line prediction script
├── validate_model.py                     ← checks the model artifact + metrics
├── rebuild_pipeline.py                   ← re-fits the pipeline on this machine
├── models/                               ← trained model files
│   ├── best_pipeline.pkl
│   ├── threshold.json
│   └── evaluation_results.json
├── reports/                              ← plots and metrics (from Kaggle output)
├── sample_test.csv                       ← 100 transactions, no fraud
├── sample_with_fraud.csv                 ← 50 transactions, 10 fraud
├── sample_predictions.csv                ← predictions for sample_test.csv
├── sample_with_fraud_predictions.csv     ← predictions for sample_with_fraud.csv
├── requirements.txt
└── README.md                             ← you are here
```

---

## 🚀 Step 1 — Train the Model on Kaggle (Free GPU)

### 1a. Create a Kaggle account
Go to [kaggle.com](https://www.kaggle.com) and sign up (it's free).

### 1b. Upload the notebook
1. Go to [kaggle.com/code](https://www.kaggle.com/code) → **"New Notebook"**
2. Click **File → Import Notebook**
3. Upload `notebook/fraud_detection_training.ipynb`

### 1c. Add the dataset
1. On the right panel, click **"+ Add data"**
2. Search for **"Credit Card Fraud Detection"** (by ULB)
3. Click **"Add"**

### 1d. Enable GPU
1. On the right panel, click **"Session options"**
2. Set **Accelerator** to **"GPU T4 x2"** or **"GPU P100"**
3. Click **"Save"**

### 1e. Run the notebook
Click **"Run All"** (the ▶▶ button at the top).  
The full run takes **~15–30 minutes**.

### 1f. Download the model files
When training finishes:
1. Click **"Output"** on the right panel
2. Download these files:
   - `best_pipeline.pkl`
   - `threshold.json`
   - `evaluation_results.json`
   - `model_comparison.csv`
   - All `.png` plot files
3. Place `.pkl` and `.json` files in the `models/` folder of this project
4. Place `.png` and `.csv` files in the `reports/` folder

> **Version mismatch?** A `.pkl` saved on Kaggle may not load with a different
> scikit-learn / XGBoost version (`InconsistentVersionWarning`,
> `XGBoostError: input stream corrupted`). The `models/best_pipeline.pkl` shipped
> with this project has **already been converted** from the Kaggle output, so it
> loads and scores exactly like the original run — no action needed.
> `python rebuild_pipeline.py` exists only as a fallback if you replace the
> artifact with a raw Kaggle download (see Step 5).

---

## 💻 Step 2 — Set Up Local Environment

Open a terminal / command prompt in this folder and run:

```bash
pip install -r requirements.txt
```

This installs everything needed for local inference and the Streamlit app.

---

## 🔮 Step 3 — Run the Streamlit App

```bash
streamlit run app.py
```

Your browser will open automatically at `http://localhost:8501`.

**How to use the app:**
1. Click **"Browse files"** and upload a CSV with transaction data
2. The app will score every transaction and show:
   - A confusion matrix at the top (when the CSV includes a `Class` column)
   - Total transactions, fraud count, fraud rate
   - A probability distribution chart
   - A colour-coded results table (fraud rows highlighted in red)
   - A download button for the predictions

The CSV must have columns: `V1`–`V28`, `Amount`.  
`Time` is recommended (used for display/ordering) and `Class` is optional.
When `Class` is present the app shows the confusion matrix; accuracy,
precision, recall and F1 are available from `python validate_model.py`.

---

## 🖥️ Step 4 (Alternative) — Command-Line Prediction

```bash
python predict.py --input my_transactions.csv --output results.csv
```

---

## ✅ Step 5 — Verify the Model

`validate_model.py` loads the saved pipeline and, when a `Class` column is
available, prints precision / recall / PR-AUC.

Score the bundled fraud sample (10 fraud + 40 legitimate transactions):

```bash
python validate_model.py
```

Reproduce the held-out test-set metrics from the full dataset:

```bash
python validate_model.py --test-split
```

Expected output for the fraud sample:

```
fraud cases  : 10
PR-AUC       : 1.0000
precision    : 1.0000
recall       : 1.0000
confusion matrix:
[[40  0]
 [ 0 10]]
```

---

## 📊 What the Models Learn

| Model | Imbalance Handling |
|---|---|
| Logistic Regression | `class_weight='balanced'` |
| Random Forest | `class_weight='balanced_subsample'` |
| XGBoost | `scale_pos_weight` = #legit / #fraud |

> **Primary metric: Average Precision (PR-AUC)**  
> This is the area under the Precision-Recall curve. It is much more meaningful than accuracy for heavily imbalanced datasets like this one (~0.17% fraud rate).

---

## 🔬 Dataset Facts

- **Source**: [Kaggle ULB Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- **Size**: 284,807 transactions over 2 days
- **Fraud rate**: 492 fraud cases (0.172%)
- **Features**: `Time` (seconds elapsed), `V1`–`V28` (anonymized PCA features), `Amount`
- **Target**: `Class` (1 = fraud, 0 = legitimate)

---

## ⚙️ Reproducibility

- Random seed: `42` everywhere
- Chronological (no-shuffle) 60/20/20 split prevents data leakage
- Scaler fitted only on training data
- Threshold selected on validation set, not test set
- Test set evaluated exactly **once**
- Shipped artifact is the exact Kaggle model (converted to UBJSON), so
  `validate_model.py --test-split` reproduces the numbers below exactly

---

## 📋 Results

Selected model: **XGBoost** (threshold tuned on the validation split,
`0.9997`), evaluated **once** on the held-out test set:

| Metric | Validation | Test (held-out) |
|---|---|---|
| PR-AUC (primary) | 1.0000 | 0.8006 |
| ROC-AUC | 1.0000 | 0.9791 |
| Precision | 1.0000 | 0.9778 |
| Recall | 1.0000 | 0.5946 |
| F1 | 1.0000 | 0.7395 |

Model comparison on the validation split (from `reports/model_comparison.csv`):

| Model | PR-AUC | ROC-AUC | Precision | Recall | F1 |
|---|---|---|---|---|---|
| XGBoost | 1.0000 | 1.0000 | 1.0000 | 1.0000 | 1.0000 |
| Random Forest | 0.9971 | 1.0000 | 0.9828 | 1.0000 | 0.9913 |
| Logistic Regression | 0.7755 | 0.9832 | 0.9524 | 0.7018 | 0.8081 |

> The validation split is very small (57 fraud cases), so its perfect scores are
> optimistic. The test numbers above are the honest, out-of-sample estimate:
> at the tuned threshold the model produced **1 false positive** and **missed 30
> of 74 frauds**. Lower the threshold in `models/threshold.json` if recall
> matters more than precision in your use case.

---

## 🆘 Troubleshooting

| Problem | Fix |
|---|---|
| `Model not found` in app | Download `best_pipeline.pkl` from Kaggle output → `models/` folder |
| `Missing columns` error | Your CSV must have `Time`, `V1`–`V28`, `Amount` |
| Kaggle GPU not available | Switch to CPU; XGBoost falls back automatically |
| `pip install` errors | Make sure Python 3.10+ is installed |
| `InconsistentVersionWarning` or `input stream corrupted` when loading the `.pkl` | You are using a raw Kaggle `.pkl` instead of the converted one shipped here. Either restore `models/best_pipeline.pkl` from this project, or re-fit locally with `python rebuild_pipeline.py` (results will be close but not identical) |
| App shows `'numpy.ndarray' object has no attribute 'map'` | Update to the current `app.py` / `predict.py` (fixed) |
