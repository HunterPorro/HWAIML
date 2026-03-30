"""
Legacy experiment: probabilistic (log-price) XGBoost + per-row offer via scipy (lognormal).
Uses only Cars_HW_data.csv + Cars_HW_template.xlsx (same inputs as optimize_to_810k.py).
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from xgboost import XGBRegressor
from scipy.optimize import minimize_scalar
from scipy.stats import norm

df = pd.read_csv('Cars_HW_data.csv')
template = pd.read_excel('Cars_HW_template.xlsx')

unlabeled_ids = set(template['ID'])
train_full = df[~df['ID'].isin(unlabeled_ids)].copy()
test_submit = df[df['ID'].isin(unlabeled_ids)].copy()
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

# Predict log(y)
log_y_full = np.log(y_full)

model_mu = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=1000, max_depth=10, learning_rate=0.03, subsample=0.8, colsample_bytree=0.8)
pipe_mu = Pipeline([('prep', preprocessor), ('reg', model_mu)])

model_sigma = XGBRegressor(random_state=42, n_jobs=-1, n_estimators=500, max_depth=6, learning_rate=0.05, subsample=0.8, colsample_bytree=0.8)
pipe_sigma = Pipeline([('prep', preprocessor), ('reg', model_sigma)])

kf = KFold(n_splits=5, shuffle=True, random_state=42)
oof_mu = np.zeros_like(log_y_full)
for train_idx, val_idx in kf.split(X_full):
    X_tr, X_v = X_full.iloc[train_idx], X_full.iloc[val_idx]
    y_tr = log_y_full.iloc[train_idx]
    pipe_mu.fit(X_tr, y_tr)
    oof_mu[val_idx] = pipe_mu.predict(X_v)

oof_residuals = log_y_full - oof_mu
oof_log_sq_residuals = np.log(oof_residuals**2 + 1e-8)

# Train final mu model
pipe_mu.fit(X_full, log_y_full)
# Train final sigma model on OOF squared residuals
pipe_sigma.fit(X_full, oof_log_sq_residuals)

X_test = test_submit.drop(columns=drop_cols, errors='ignore')
mu_test = pipe_mu.predict(X_test)
log_sq_sigma_test = pipe_sigma.predict(X_test)
sigma_test = np.sqrt(np.exp(log_sq_sigma_test))

def expected_profit(offer, mu, sigma):
    z_min = (np.log(offer / 0.85) - mu) / sigma
    term1 = np.exp(mu + sigma**2 / 2) * (1 - norm.cdf(z_min - sigma))
    term2 = (offer + 75) * (1 - norm.cdf(z_min))
    return term1 - term2

test_offers = []
for i in range(len(mu_test)):
    mu = mu_test[i]
    sigma = sigma_test[i]
    guess = 0.85 * np.exp(mu)
    res = minimize_scalar(lambda o: -expected_profit(o, mu, sigma), bounds=(guess * 0.5, guess * 1.5), method='bounded')
    if res.success:
        test_offers.append(res.x)
    else:
        test_offers.append(guess)

test_offers = np.array(test_offers)

template['BestGuessAtPrice'] = np.exp(mu_test)
template['OfferPrice'] = test_offers
template.to_excel('Cars_HW_template_predictions.xlsx', index=False)
print("Saved predictions to Cars_HW_template_predictions.xlsx")
