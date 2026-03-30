"""
Used-car offer optimizer — clean pipeline.
  Rule: if offer >= 0.85 * true_value -> buy; profit = true_value - offer - 75
  Goal: maximize total profit across 8,531 template cars.
"""
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
from sklearn.metrics import mean_squared_error
import lightgbm as lgb
from xgboost import XGBRegressor

TARGET_PROFIT = 610_000
SEED = 42
np.random.seed(SEED)

def log(msg):
    print(msg, flush=True)

def compute_profit(actual, offer):
    actual, offer = np.asarray(actual, float), np.asarray(offer, float)
    won = offer >= 0.85 * actual
    return float(np.where(won, actual - offer - 75, 0).sum()), int(won.sum())

# ── 1. Load (with /tmp caching for iCloud-evicted files) ──────────────────────
import os, pickle

DATA_DIR = '/Users/hunterporro/Desktop/Applied ML HW/HW 4'
CACHE_DIR = '/tmp/hw4_cache'
os.makedirs(CACHE_DIR, exist_ok=True)

def load_with_cache(filename, reader, **kw):
    cache_path = os.path.join(CACHE_DIR, filename + '.pkl')
    if os.path.exists(cache_path):
        log(f"  Loading {filename} from cache...")
        return pickle.load(open(cache_path, 'rb'))
    log(f"  Loading {filename} from disk...")
    data = reader(os.path.join(DATA_DIR, filename), **kw)
    pickle.dump(data, open(cache_path, 'wb'))
    return data

log("Loading data...")
df = load_with_cache('Cars_HW_data.csv', pd.read_csv)
template = load_with_cache('Cars_HW_template.xlsx', pd.read_excel)
tmpl_ids = set(template['ID'])
train_df = df[~df['ID'].isin(tmpl_ids)].copy()
test_df  = df[df['ID'].isin(tmpl_ids)].copy()
log(f"Train: {len(train_df):,}  Test: {len(test_df):,}")

# ── 2. Features ───────────────────────────────────────────────────────────────
def add_features(d):
    d = d.copy()
    d['Car_Age'] = 2026 - d['Year']
    d['Log_Odometer'] = np.log1p(d['Odometer_KM'])
    d['Odo_per_Year'] = d['Odometer_KM'] / np.maximum(d['Car_Age'], 1)
    d['Age_x_LogOdo'] = d['Car_Age'] * d['Log_Odometer']
    d['Log_Engine'] = np.log1p(d['Engine_Capacity'].fillna(0))
    d['Age_x_Engine'] = d['Car_Age'] * d['Log_Engine']
    d['Odo_x_Engine'] = d['Log_Odometer'] * d['Log_Engine']
    d['Car_Age_sq'] = d['Car_Age'] ** 2
    d['Log_Odo_sq'] = d['Log_Odometer'] ** 2
    d['Photos_per_Day'] = d['Photos_In_Listing'] / np.maximum(d['Days_Listed'], 1)
    return d

train_df = add_features(train_df)
test_df  = add_features(test_df)

# ── 3. Target encoding ───────────────────────────────────────────────────────
def target_encode(train, test, col, target='Price', n_folds=5, smoothing=50):
    global_mean = train[target].mean()
    te_col = f'{col}_te'
    train[te_col] = global_mean
    kf = KFold(n_splits=n_folds, shuffle=True, random_state=SEED)
    for tr_idx, val_idx in kf.split(train):
        grp = train.iloc[tr_idx].groupby(col)[target]
        means = grp.mean()
        counts = grp.count()
        smoothed = (means * counts + global_mean * smoothing) / (counts + smoothing)
        train.iloc[val_idx, train.columns.get_loc(te_col)] = (
            train.iloc[val_idx][col].map(smoothed).fillna(global_mean).values)
    grp = train.groupby(col)[target]
    means = grp.mean(); counts = grp.count()
    smoothed = (means * counts + global_mean * smoothing) / (counts + smoothing)
    test[te_col] = test[col].map(smoothed).fillna(global_mean)
    return train, test

for col in ['Make', 'Model']:
    train_df, test_df = target_encode(train_df, test_df, col)

train_df['Make_Model'] = train_df['Make'] + '_' + train_df['Model']
test_df['Make_Model']  = test_df['Make']  + '_' + test_df['Model']
train_df, test_df = target_encode(train_df, test_df, 'Make_Model')

# Count encoding
for col in ['Make', 'Model', 'Make_Model']:
    counts = train_df[col].value_counts().to_dict()
    train_df[f'{col}_count'] = train_df[col].map(counts)
    test_df[f'{col}_count']  = test_df[col].map(counts).fillna(0)

log(f"Features engineered.")

# ── 4. Feature matrix ─────────────────────────────────────────────────────────
cat_cols = ['Transmission', 'Color', 'Fuel Type', 'Engine_Type', 'Body',
            'Condition', 'Drivetrain']
bool_cols = [f'feature_{i}' for i in range(10)] + ['Warranty']
num_cols = ['Odometer_KM', 'Engine_Capacity', 'Photos_In_Listing',
            'Profile_Likes', 'Days_Listed',
            'Car_Age', 'Log_Odometer', 'Odo_per_Year', 'Age_x_LogOdo',
            'Log_Engine', 'Age_x_Engine', 'Odo_x_Engine',
            'Car_Age_sq', 'Log_Odo_sq', 'Photos_per_Day',
            'Make_te', 'Model_te', 'Make_Model_te',
            'Make_count', 'Model_count', 'Make_Model_count']
feature_cols = num_cols + bool_cols + cat_cols

for c in cat_cols:
    train_df[c] = train_df[c].astype('category')
    test_df[c]  = test_df[c].astype('category')
for c in bool_cols:
    train_df[c] = train_df[c].astype(int)
    test_df[c]  = test_df[c].astype(int)

X_train = train_df[feature_cols]
y_train = train_df['Price'].values
y_train_log = np.log1p(y_train)
X_test  = test_df[feature_cols]
log(f"Features: {len(feature_cols)} columns")

# ── 5. LightGBM on log(price) — main model ───────────────────────────────────
N_FOLDS = 5
oof_lgb_log = np.zeros(len(X_train))
test_lgb_log_list = []

lgb_params_log = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'learning_rate': 0.015, 'num_leaves': 255, 'max_depth': -1,
    'min_child_samples': 15, 'feature_fraction': 0.75, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'lambda_l1': 0.1, 'lambda_l2': 3.0,
    'verbosity': -1, 'seed': SEED, 'n_jobs': -1,
}

kf = KFold(n_splits=N_FOLDS, shuffle=True, random_state=SEED)
for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    log(f"LGB-log fold {fold+1}/{N_FOLDS}...")
    dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train_log[tr_idx], categorical_feature=cat_cols)
    dval   = lgb.Dataset(X_train.iloc[val_idx], label=y_train_log[val_idx], categorical_feature=cat_cols, reference=dtrain)
    model = lgb.train(lgb_params_log, dtrain, num_boost_round=5000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_lgb_log[val_idx] = np.expm1(model.predict(X_train.iloc[val_idx]))
    test_lgb_log_list.append(np.expm1(model.predict(X_test)))

rmse_lgb_log = np.sqrt(mean_squared_error(y_train, oof_lgb_log))
log(f"LGB-log OOF RMSE: {rmse_lgb_log:,.2f}")
test_lgb_log = np.mean(test_lgb_log_list, axis=0)

# ── 6. LightGBM on raw price — secondary model ───────────────────────────────
oof_lgb_raw = np.zeros(len(X_train))
test_lgb_raw_list = []

lgb_params_raw = {
    'objective': 'regression', 'metric': 'rmse', 'boosting_type': 'gbdt',
    'learning_rate': 0.015, 'num_leaves': 200, 'max_depth': -1,
    'min_child_samples': 20, 'feature_fraction': 0.8, 'bagging_fraction': 0.8,
    'bagging_freq': 5, 'lambda_l1': 0.1, 'lambda_l2': 5.0,
    'verbosity': -1, 'seed': SEED, 'n_jobs': -1,
}

for fold, (tr_idx, val_idx) in enumerate(kf.split(X_train)):
    log(f"LGB-raw fold {fold+1}/{N_FOLDS}...")
    dtrain = lgb.Dataset(X_train.iloc[tr_idx], label=y_train[tr_idx], categorical_feature=cat_cols)
    dval   = lgb.Dataset(X_train.iloc[val_idx], label=y_train[val_idx], categorical_feature=cat_cols, reference=dtrain)
    model = lgb.train(lgb_params_raw, dtrain, num_boost_round=5000,
                      valid_sets=[dval], callbacks=[lgb.early_stopping(100), lgb.log_evaluation(0)])
    oof_lgb_raw[val_idx] = model.predict(X_train.iloc[val_idx])
    test_lgb_raw_list.append(model.predict(X_test))

rmse_lgb_raw = np.sqrt(mean_squared_error(y_train, oof_lgb_raw))
log(f"LGB-raw OOF RMSE: {rmse_lgb_raw:,.2f}")
test_lgb_raw = np.mean(test_lgb_raw_list, axis=0)

# ── 7. XGBoost on log(price) — third model ───────────────────────────────────
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline as SKPipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import StandardScaler, OrdinalEncoder

xgb_preproc = ColumnTransformer([
    ('num', SKPipeline([('imp', SimpleImputer(strategy='median')),
                        ('sc', StandardScaler())]), num_cols + bool_cols),
    ('cat', SKPipeline([('imp', SimpleImputer(strategy='constant', fill_value='missing')),
                        ('ord', OrdinalEncoder(handle_unknown='use_encoded_value',
                                               unknown_value=-1))]), cat_cols),
])

oof_xgb = np.zeros(len(X_train))
test_xgb_list = []
XGB_SEEDS = [42, 43, 44]

for seed in XGB_SEEDS:
    kf2 = KFold(n_splits=N_FOLDS, shuffle=True, random_state=seed)
    for fold, (tr_idx, val_idx) in enumerate(kf2.split(X_train)):
        Xp_tr = xgb_preproc.fit_transform(X_train.iloc[tr_idx])
        Xp_val = xgb_preproc.transform(X_train.iloc[val_idx])
        Xp_test = xgb_preproc.transform(X_test)
        m = XGBRegressor(n_estimators=2000, max_depth=7, learning_rate=0.02,
                         subsample=0.8, colsample_bytree=0.8, reg_lambda=5, reg_alpha=0.5,
                         random_state=seed, n_jobs=1, early_stopping_rounds=50, eval_metric='rmse')
        m.fit(Xp_tr, y_train_log[tr_idx], eval_set=[(Xp_val, y_train_log[val_idx])], verbose=False)
        oof_xgb[val_idx] += np.expm1(m.predict(Xp_val)) / len(XGB_SEEDS)
        test_xgb_list.append(np.expm1(m.predict(Xp_test)))
    log(f"XGB seed {seed} done.")

rmse_xgb = np.sqrt(mean_squared_error(y_train, oof_xgb))
log(f"XGB OOF RMSE: {rmse_xgb:,.2f}")
test_xgb = np.mean(test_xgb_list, axis=0)

# ── 8. Optimal blend ─────────────────────────────────────────────────────────
log("\nFinding optimal blend weights...")
models_oof = [oof_lgb_log, oof_lgb_raw, oof_xgb]
models_test = [test_lgb_log, test_lgb_raw, test_xgb]
model_names = ['LGB-log', 'LGB-raw', 'XGB']

best_blend_rmse = np.inf
best_w = [1/3, 1/3, 1/3]
for w0 in np.linspace(0, 1, 21):
    for w1 in np.linspace(0, 1 - w0, max(1, int((1 - w0) * 20) + 1)):
        w2 = 1 - w0 - w1
        if w2 < 0: continue
        blend = w0 * models_oof[0] + w1 * models_oof[1] + w2 * models_oof[2]
        rmse = np.sqrt(mean_squared_error(y_train, blend))
        if rmse < best_blend_rmse:
            best_blend_rmse = rmse
            best_w = [w0, w1, w2]

for i, n in enumerate(model_names):
    log(f"  {n} weight: {best_w[i]:.2f}")
log(f"Blended OOF RMSE: {best_blend_rmse:,.2f}")

final_oof  = sum(w * p for w, p in zip(best_w, models_oof))
final_test = sum(w * p for w, p in zip(best_w, models_test))

# ── 9. Offer optimization: threshold + multiplier ────────────────────────────
log("\nOptimizing offer strategy (threshold + multiplier)...")

# Key insight: cheap cars have large relative error → lose money.
# Only bid on cars above a predicted-price threshold.
best_total_profit = -np.inf
best_config = (0, 0.85)

for threshold in np.linspace(0, 8000, 81):
    for mult in np.linspace(0.840, 0.920, 81):
        offers = np.where(final_oof > threshold, final_oof * mult, 0)
        prof, wins = compute_profit(y_train, offers)
        if prof > best_total_profit:
            best_total_profit = prof
            best_config = (threshold, mult)

threshold_opt, mult_opt = best_config
_, best_wins = compute_profit(y_train, np.where(final_oof > threshold_opt, final_oof * mult_opt, 0))
n_bid = (final_oof > threshold_opt).sum()
log(f"Best: threshold=${threshold_opt:,.0f}  mult={mult_opt:.4f}")
log(f"  OOF profit: ${best_total_profit:,.2f}  wins: {best_wins}/{n_bid} bids ({100*best_wins/max(n_bid,1):.1f}%)")

scaled = (best_total_profit / len(y_train)) * len(template)
log(f"  Scaled expected profit: ${scaled:,.2f}")

# Also try per-decile (only for cars above threshold)
log("\nPer-decile optimization (above threshold)...")
above = final_oof > threshold_opt
oof_above = final_oof[above]
y_above = y_train[above]

if len(oof_above) > 0:
    n_dec = min(10, max(2, len(oof_above) // 500))
    dec_bounds = np.percentile(oof_above, np.linspace(0, 100, n_dec + 1)[1:-1])
    dec_mults = np.zeros(n_dec)
    dec_total = 0.0

    for d in range(n_dec):
        if d == 0:
            mask = oof_above <= dec_bounds[0]
        elif d == n_dec - 1:
            mask = oof_above > dec_bounds[-1]
        else:
            mask = (oof_above > dec_bounds[d-1]) & (oof_above <= dec_bounds[d])
        y_d, p_d = y_above[mask], oof_above[mask]
        best_dm, best_dp = mult_opt, -np.inf
        for m in np.linspace(0.830, 0.930, 101):
            pr, _ = compute_profit(y_d, p_d * m)
            if pr > best_dp:
                best_dp, best_dm = pr, m
        dec_mults[d] = best_dm
        dec_total += best_dp
        _, dw = compute_profit(y_d, p_d * best_dm)
        log(f"  Decile {d+1}: mult={best_dm:.3f}  profit=${best_dp:,.0f}  wins={dw}/{len(y_d)}")

    dec_scaled = (dec_total / len(y_train)) * len(template)
    log(f"  Decile scaled expected: ${dec_scaled:,.2f}")
    use_decile = dec_total > best_total_profit
else:
    use_decile = False
    dec_scaled = 0

if use_decile:
    log("Using per-decile multipliers.")
    estimated_profit = dec_scaled
else:
    log("Using global threshold + multiplier.")
    estimated_profit = scaled

meets = estimated_profit >= TARGET_PROFIT
log(f"\n{'='*60}")
log(f"Expected profit: ${estimated_profit:,.2f}  Target: ${TARGET_PROFIT:,.2f}  Met: {'YES' if meets else 'NO'}")
log(f"{'='*60}")

# ── 10. Build test offers ─────────────────────────────────────────────────────
if use_decile:
    test_above = final_test > threshold_opt
    test_offers = np.zeros(len(final_test))
    for d in range(n_dec):
        if d == 0:
            mask = test_above & (final_test <= dec_bounds[0])
        elif d == n_dec - 1:
            mask = test_above & (final_test > dec_bounds[-1])
        else:
            mask = test_above & (final_test > dec_bounds[d-1]) & (final_test <= dec_bounds[d])
        test_offers[mask] = final_test[mask] * dec_mults[d]
    leftover = test_above & (test_offers == 0)
    test_offers[leftover] = final_test[leftover] * mult_opt
else:
    test_offers = np.where(final_test > threshold_opt, final_test * mult_opt, 0)

n_bids_test = (test_offers > 0).sum()
log(f"Test: {n_bids_test}/{len(test_offers)} cars get bids  (skip {len(test_offers)-n_bids_test})")
log(f"OfferPrice: min={test_offers[test_offers>0].min():,.0f}  median={np.median(test_offers[test_offers>0]):,.0f}  max={test_offers.max():,.0f}")

# ── 11. Export ────────────────────────────────────────────────────────────────
out = test_df[['ID']].copy()
out['BestGuessAtPrice'] = final_test
out['OfferPrice'] = test_offers
result = template[['ID']].merge(out, on='ID', how='left')
assert result['BestGuessAtPrice'].isnull().sum() == 0
result.to_excel('Cars_HW_template_predictions.xlsx', index=False)
log("Exported Cars_HW_template_predictions.xlsx")

with open('expected_profit.txt', 'w') as f:
    f.write(f"Estimated submission profit (scaled from OOF): ${estimated_profit:,.2f}\n")
    f.write(f"Target: ${TARGET_PROFIT:,.2f}\n")
    f.write(f"Target met: {'YES' if meets else 'NO'}\n")
    f.write(f"OOF RMSE (blended): {best_blend_rmse:,.2f}\n")
    f.write(f"Threshold: ${threshold_opt:,.0f}  Multiplier: {mult_opt:.4f}\n")
log("Done.")
