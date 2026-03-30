"""
HW-only pipeline: ensemble of 3 XGBoost models + per-bin (m, o) grid search.
Saves BestGuessAtPrice and OfferPrice to Cars_HW_template_predictions.xlsx.
"""
import sys
sys.stdout.reconfigure(line_buffering=True)

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor

df = pd.read_csv('Cars_HW_data.csv')
template = pd.read_excel('Cars_HW_template.xlsx')

unlabeled_ids = set(template['ID'])
train_full = df[~df['ID'].isin(unlabeled_ids)].copy()
test_submit = df[df['ID'].isin(unlabeled_ids)].copy()
# Row i must match template row i (predictions assigned by position)
assert set(test_submit['ID']) == set(template['ID']), "Every template ID must appear in Cars_HW_data.csv"
test_submit = test_submit.set_index('ID').reindex(template['ID']).reset_index()

train_full['Year'] = train_full['Year'].astype(float)
test_submit['Year'] = test_submit['Year'].astype(float)
train_full['Car_Age'] = 2026 - train_full['Year']
test_submit['Car_Age'] = 2026 - test_submit['Year']

drop_cols = ['ID', 'Price', 'Year']
X_full = train_full.drop(columns=drop_cols, errors='ignore')
y_full = train_full['Price']

num_cols = X_full.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_full.select_dtypes(include=['object', 'bool', 'string']).columns
for c in cat_cols:
    X_full[c] = X_full[c].astype(str)
    test_submit[c] = test_submit[c].astype(str)

num_transformer = Pipeline([('imputer', SimpleImputer(strategy='median')), ('scaler', StandardScaler())])
cat_transformer = Pipeline([('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
                           ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))])
preprocessor = ColumnTransformer([('num', num_transformer, num_cols), ('cat', cat_transformer, cat_cols)])

def compute_profit(actual, offer):
    actual, offer = np.array(actual), np.array(offer)
    accepted = offer >= 0.85 * actual
    profit = np.where(accepted, actual - offer - 75, 0)
    return profit.sum(), accepted.sum()

X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)

# Ensemble: 3 models with different seeds, average predictions
val_preds_list = []
pipes = []
for seed in [42, 43, 44]:
    model = XGBRegressor(random_state=seed, n_jobs=-1, n_estimators=1000, max_depth=10, learning_rate=0.03,
                         subsample=0.8, colsample_bytree=0.8)
    pipe = Pipeline([('prep', preprocessor), ('reg', model)])
    pipe.fit(X_train, y_train)
    val_preds_list.append(pipe.predict(X_val))
    pipes.append(pipe)

val_preds = np.mean(val_preds_list, axis=0)

bins = [0, 3000, 6000, 10000, 15000, 25000, np.inf]
best_params_per_bin = []
total_val_profit = 0

for i in range(len(bins) - 1):
    mask = (val_preds >= bins[i]) & (val_preds < bins[i + 1])
    if not np.any(mask):
        best_params_per_bin.append((1.0, 0))
        continue
    y_val_bin = y_val[mask]
    val_preds_bin = val_preds[mask]
    best_p = -np.inf
    best_m = 0.85
    best_o = 0.0
    if i == 0:
        m_range = np.linspace(0.80, 0.90, 60)
        o_range = np.linspace(1500, 2500, 60)
    elif i == 1:
        m_range = np.linspace(0.75, 0.85, 60)
        o_range = np.linspace(1500, 2500, 60)
    elif i == 2:
        m_range = np.linspace(0.82, 0.92, 60)
        o_range = np.linspace(-200, 1200, 60)
    elif i == 3:
        m_range = np.linspace(0.75, 0.85, 60)
        o_range = np.linspace(-800, 600, 60)
    elif i == 4:
        m_range = np.linspace(0.75, 0.85, 60)
        o_range = np.linspace(-1200, 200, 60)
    else:
        m_range = np.linspace(0.80, 0.95, 60)
        o_range = np.linspace(500, 2200, 60)
    for m in m_range:
        for o in o_range:
            p, _ = compute_profit(y_val_bin, val_preds_bin * m - o)
            if p > best_p:
                best_p = p
                best_m = m
                best_o = o
    best_params_per_bin.append((best_m, best_o))
    total_val_profit += best_p
    print(f"Bin {i+1} best m={best_m:.4f}, offset={best_o:.2f} with profit {best_p:,.2f}")

print(f"Total Val Profit ${total_val_profit:,.2f}")

X_test = test_submit.drop(columns=drop_cols, errors='ignore')
test_preds_list = []
for pipe in pipes:
    pipe.fit(X_full, y_full)
    test_preds_list.append(pipe.predict(X_test))
test_preds = np.mean(test_preds_list, axis=0)

test_offers = np.zeros_like(test_preds)
for i in range(len(bins) - 1):
    mask = (test_preds >= bins[i]) & (test_preds < bins[i + 1])
    m, o = best_params_per_bin[i]
    test_offers[mask] = test_preds[mask] * m - o

assert len(test_offers) == len(template), "test_offers length must match template"
assert np.isfinite(test_preds).all() and np.isfinite(test_offers).all(), "No NaN/Inf in predictions or offers"

template['BestGuessAtPrice'] = test_preds
template['OfferPrice'] = test_offers
template.to_excel('Cars_HW_template_predictions.xlsx', index=False)
print("Saved predictions to Cars_HW_template_predictions.xlsx")
