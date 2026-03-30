# HW 4 — Code walkthrough and project summary

This document explains **what each part of `optimize_to_810k.py` does** and summarizes **the work done** on this assignment (modeling, cleanup, and deliverables).

---

## Part A — What each part of `optimize_to_810k.py` does

The script runs top to bottom in one pass: load data → train models → tune offers on validation → refit on all training data → predict for template rows → save Excel.

| Lines (approx.) | What it does |
|-----------------|--------------|
| **1–15** | Imports: `pandas`/`numpy`, sklearn preprocessing and `train_test_split`, `Pipeline`, `XGBRegressor`. `stdout` line-buffering so prints show up promptly. |
| **17–18** | Loads **`Cars_HW_data.csv`** (all cars) and **`Cars_HW_template.xlsx`** (IDs you must submit predictions for). |
| **20–25** | **Train vs “test” split:** rows whose `ID` is **in** the template are the submission set; all other rows are **training** (they have known `Price`). `test_submit` is **reindexed** to match **`template['ID']` order** so row *i* of predictions matches row *i* of the template (not random CSV order). Assert ensures every template ID exists in the CSV. |
| **27–30** | **Feature engineering:** `Year` as float; **`Car_Age = 2026 − Year`** for train and submission rows. |
| **32–40** | **Features and labels:** `X_full` = training features (drops `ID`, `Price`, raw `Year` — `Car_Age` stays). `y_full` = `Price`. Categorical columns are cast to **strings** so one-hot encoding is stable. |
| **42–45** | **Preprocessor:** numeric columns → median imputation + `StandardScaler`; categorical → fill `"missing"` + **one-hot** (`handle_unknown='ignore'`). Bundled in a `ColumnTransformer`. |
| **47–51** | **`compute_profit(actual, offer)`:** Implements the assignment rule — a deal **wins** if `offer >= 0.85 × actual`; profit per win is `actual − offer − 75`; otherwise profit 0. Returns total profit and count of wins. |
| **53** | **Validation split:** 80% train / 20% validation from **labeled training rows only** (`random_state=42`). Used only to tune offer parameters `(m, o)`, not to pick hyperparameters of XGBoost in this script. |
| **55–66** | **Ensemble:** Trains **three** identical XGBoost pipelines (seeds **42, 43, 44**): 1000 trees, `max_depth=10`, `learning_rate=0.03`, row/column subsample 0.8. Each fits on `X_train, y_train`. **Validation predictions** are the **average** of the three models’ predictions on `X_val`. |
| **68–111** | **Per-bin offer tuning (grid search on validation):** Predicted price is assigned to **bins** (e.g. &lt;$3k, $3k–$6k, …). For each bin, search **multiplier `m`** and **offset `o`** so **`offer = m × predicted_price − o`**. Ranges for `m` and `o` differ by bin. Objective: **maximize total validation profit** using `compute_profit` on **validation actual prices** (`y_val`) and synthetic offers. Stores best `(m, o)` per bin and prints per-bin and total validation profit. |
| **113–118** | **Refit on all labeled training rows:** Each of the three pipelines is fit on `X_full, y_full`. Average their predictions on **`X_test`** (submission features) → **`test_preds`** (ensemble **best-guess price**). |
| **120–124** | **Build submission offers:** Same bin edges as above; for each submission row, assign bin from **`test_preds`**, apply that bin’s `(m, o)` → **`test_offers`**. |
| **126–132** | **Checks and save:** Length matches template; no NaN/Inf. Writes **`BestGuessAtPrice`** and **`OfferPrice`** into the template dataframe and saves **`Cars_HW_template_predictions.xlsx`**. |

**Alternate script — `optimize_profit30.py`:** Same data logic and preprocessing, but **one** XGBoost model (seed 42) instead of a 3-model average; **finer grids** (100×100) per bin on validation. Same output file name when run.

---

## Part B — Summary of work done (chronologically / thematically)

1. **Modeling approach**
   - **Regression:** XGBoost on tabular features with mixed numeric + categorical inputs.
   - **Ensemble:** Three seeds averaged to reduce variance in price predictions.
   - **Offer layer:** Not a single global rule — **price-bin–specific** linear transforms of predictions, tuned on a **validation** split to maximize **profit** under the 0.85 acceptance rule and $75 fee, not raw RMSE alone.
   - **Submission alignment:** Test rows ordered to match the template by `ID` so the Excel file is correct row-for-row.

2. **Iteration and targets**
   - Earlier versions experimented with stronger models, scipy-based optimizers, and **blending** predictions from multiple sources.
   - A **profit target** (e.g. $810k) was used during development; the **clean pipeline** always writes predictions after a successful run.

3. **Cleanup for a “standalone” submission**
   - Refocused the **main** scripts on the required inputs only.
   - **Canonical entry points:** `optimize_to_810k.py` (primary), `optimize_profit30.py` (alternate).
   - Legacy / experimental scripts were moved to **`legacy_archive_not_for_submission/`** so the main folder stays focused.
   - **`README.md`** documents inputs, command to run, and high-level logic.

4. **Deliverables**
   - **`Cars_HW_template_predictions.xlsx`:** Filled with **`BestGuessAtPrice`** and **`OfferPrice`** for each template `ID`, produced by running the primary script on **`Cars_HW_data.csv`** + **`Cars_HW_template.xlsx`** only.

---

## Quick reference

| File | Role |
|------|------|
| `optimize_to_810k.py` | Main pipeline: 3-model ensemble + per-bin offer tuning → predictions Excel |
| `optimize_profit30.py` | Simpler variant: single XGBoost + same bin logic |
| `Cars_HW_data.csv` | All listings; training labels = `Price` where not in template |
| `Cars_HW_template.xlsx` | Submission IDs (and columns to fill) |
| `Cars_HW_template_predictions.xlsx` | Output submission file |
| `README.md` | How to run and what inputs/outputs are |
| `HW4_WORK_SUMMARY.md` | This document |
| `legacy_archive_not_for_submission/` | Old experiments; not part of the clean submission path |

---

*Generated to document the HW 4 used-car pricing and offer optimization workflow.*
