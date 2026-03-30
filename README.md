# HW 4 — Used car pricing & offers (submission pipeline)

## What to run

**Primary script:** `optimize_to_810k.py`

```bash
python3 optimize_to_810k.py
```

**Inputs (only these):**

- `Cars_HW_data.csv`
- `Cars_HW_template.xlsx`

**Output:**

- `Cars_HW_template_predictions.xlsx` — columns include `BestGuessAtPrice` and `OfferPrice`

**Alternate (single XGBoost instead of 3-model ensemble):** `optimize_profit30.py`

## Logic (standalone)

1. Train rows = cars in `Cars_HW_data.csv` whose `ID` is **not** in the template.
2. Test rows = cars whose `ID` **is** in the template; rows are **reordered** to match template order so each prediction lines up with the correct `ID`.
3. Three XGBoost regressors (seeds 42, 43, 44) average price predictions; a held-out **validation** split from train only is used to grid-search per-bin multipliers and offsets for offers (`offer = pred * m - o`) under the rule: accept if `offer >= 0.85 * actual` on validation, profit `actual - offer - 75`.
4. Models are refit on all train rows; test predictions and offers are written to the output workbook.

## Documentation

- **`HW4_WORK_SUMMARY.md`** — section-by-section explanation of `optimize_to_810k.py` and a narrative summary of the project work.

## Other files

- `legacy_archive_not_for_submission/` — older experiments; **not** part of the clean pipeline. Do not use for grading unless your instructor allows it.
- Other `.py` helpers in this folder (e.g. `run_eval.py`, `train_and_predict.py`) are separate experiments and are **not** required for the submission script above.
