# models/

Artifacts used by `predict.py`, `validate_model.py` and `app.py`.

| File | Description |
|---|---|
| `best_pipeline.pkl` | `StandardScaler` + tuned `XGBClassifier` (scikit-learn `Pipeline`) |
| `threshold.json` | Optimised decision threshold (`{"threshold": 0.9997, "model": "XGBoost"}`) |
| `evaluation_results.json` | Full test-set and validation comparison metrics |

## About this artifact

`best_pipeline.pkl` is **the exact model trained on Kaggle** — not a re-fit.
The original Kaggle pickle was serialised with scikit-learn 1.6.1 and an older
XGBoost, which newer XGBoost versions (3.x) refuse to load
(`XGBoostError: input stream corrupted`).

The shipped file was therefore rebuilt from the Kaggle model's own contents:
the `StandardScaler` parameters and the XGBoost booster exported to UBJSON.
Predictions are **bit-identical** to the Kaggle run (`max |proba diff| = 0.0`),
and only the serialisation format changed.

No action is needed — `joblib.load("models/best_pipeline.pkl")` works on the
current environment.

## Fallback: re-fit locally

If you ever replace the artifact with a raw Kaggle output and it fails to load,
re-fit the same pipeline on this machine:

```bash
python rebuild_pipeline.py
python validate_model.py --test-split
```

`rebuild_pipeline.py` uses the same chronological split, preprocessing and
tuned hyperparameters from Kaggle, so results are very close — but a re-fit is
**not** bit-identical to the original training run (small stochastic
differences, e.g. PR-AUC ≈ 0.795 vs 0.801). Prefer the shipped artifact.

## Verified metrics

Selected model: XGBoost, threshold `0.9996687`.
Measured on the chronological held-out test split (56,746 transactions, 74 fraud).

| Metric | Validation | Test (held-out) |
|---|---|---|
| PR-AUC | 1.0000 | 0.8006 |
| ROC-AUC | 1.0000 | 0.9791 |
| Precision | 1.0000 | 0.9778 |
| Recall | 1.0000 | 0.5946 |
| F1 | 1.0000 | 0.7395 |

Confusion matrix (test):

```
[[56671     1]
 [   30    44]]
```

The validation split contains only 57 fraud cases, so its perfect scores are
optimistic. The test column is the honest out-of-sample estimate: 1 false
positive and 30 missed frauds at the tuned threshold.

Reproduce with:

```bash
python validate_model.py --test-split
```
