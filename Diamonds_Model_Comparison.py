#!/usr/bin/env python
# coding: utf-8

# # Used Car Pricing Strategy: Profit Maximization Model
# 
# ## 1. Executive Summary
# This document outlines the predictive modeling and decision optimization strategy designed to maximize profitability for the used car purchasing process. 
# 
# **Objective**: Develop an optimized bidding strategy to maximize dealer profit. 
# 
# **Strategic Approach**:
# 1. **Predictive Analytics (Machine Learning)**: Predict the true market value of the car (BestGuessAtPrice) using an ensemble of models (Forward Selection, Logistic Regression, CART, RF, AdaBoost, XGBoost).
# 2. **Offer Optimization (Prescriptive Analytics)**: Calibrate the theoretical Offer Price as a percentage coefficient of the predicted value.
# 
# The target goal is to exceed an expected profit of **$608,680.76** on template submission rows (per assignment).
# ---
# 

# In[ ]:


import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

# McKinsey/BCG Style Formatting Configuration
mc_kinsey_blue = "#003B4F"
bcg_green = "#006656"
accent_orange = "#F37021"
neutral_grey = "#A9A9A9"

plt.rcParams.update({
    'font.family': 'sans-serif',
    'font.sans-serif': ['Arial', 'Helvetica', 'DejaVu Sans'],
    'axes.prop_cycle': plt.cycler('color', [mc_kinsey_blue, bcg_green, accent_orange, neutral_grey, '#ADD8E6']),
    'axes.titleweight': 'bold',
    'axes.titlesize': 14,
    'axes.spines.top': False,
    'axes.spines.right': False,
    'axes.grid': True,
    'grid.color': '#E5E5E5',
    'grid.linestyle': '--',
    'grid.alpha': 0.7
})

def setup_mckinsey_style_axis(ax, title, xlabel, ylabel):
    ax.set_title(title, loc='left', pad=20)
    ax.set_xlabel(xlabel, fontweight='bold', color='#4A4A4A')
    ax.set_ylabel(ylabel, fontweight='bold', color='#4A4A4A')
    ax.tick_params(colors='#4A4A4A')


# ## 2. Data Ingestion & Preprocessing

# In[ ]:


print("Loading dataset...")
df = pd.read_csv('Cars_HW_data.csv')
template = pd.read_excel('Cars_HW_template.xlsx')

unlabeled_ids = set(template['ID'])
train_full = df[~df['ID'].isin(unlabeled_ids)].copy()
test_submit = df[df['ID'].isin(unlabeled_ids)].copy()

test_submit.set_index('ID', inplace=True)
test_submit = test_submit.reindex(template['ID']).reset_index()

# Feature Engineering: Age of car (Assuming cur year 2026)
train_full['Car_Age'] = 2026 - train_full['Year']
test_submit['Car_Age'] = 2026 - test_submit['Year']

drop_cols = ['ID', 'Price', 'Year']
if 'OfferPrice' in train_full.columns: drop_cols.append('OfferPrice')
if 'BestGuessAtPrice' in train_full.columns: drop_cols.append('BestGuessAtPrice')

X_full = train_full.drop(columns=drop_cols)
y_full = train_full['Price']

num_cols = X_full.select_dtypes(include=['int64', 'float64']).columns
cat_cols = X_full.select_dtypes(include=['object', 'bool', 'string']).columns
for c in cat_cols: X_full[c]=X_full[c].astype(str); test_submit[c]=test_submit[c].astype(str)

num_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])
cat_transformer = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor = ColumnTransformer(
    transformers=[
        ('num', num_transformer, num_cols),
        ('cat', cat_transformer, cat_cols)
    ])

print(f"Training instances: {X_full.shape[0]:,}")
print(f"Deployment instances: {test_submit.shape[0]:,}")


# ## 3. Advanced Modeling Pipeline & Model Selection
# 
# We rigorously benchmark a suite of regression models evaluating 5-10 parameter combinations per model using 5-Fold Cross Validation. We also evaluate the models using AUC (treating Price > Median as the positive class), as explicitly requested in the problem statement procedure.

# In[ ]:


from sklearn.decomposition import PCA
# Add PCA to handle collinearity among the many encoded categorical features / numerical features
# PCA(n_components=0.95) retains 95% of variance while removing collinear/redundant components.

num_transformer_pca = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='median')),
    ('scaler', StandardScaler())
])

cat_transformer_pca = Pipeline(steps=[
    ('imputer', SimpleImputer(strategy='constant', fill_value='missing')),
    ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
])

preprocessor_pca = ColumnTransformer(
    transformers=[
        ('num', num_transformer_pca, num_cols),
        ('cat', cat_transformer_pca, cat_cols)
    ])

preprocessor_with_pca = Pipeline(steps=[
    ('prep', preprocessor_pca),
    ('pca', PCA(n_components=0.95, random_state=42))
])

class LogisticRegressorProxy(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.model_ = None
        self.median_ = 0
        self.max_price_ = 0
        self.is_fitted_ = False

    def fit(self, X, y):
        self.model_ = LogisticRegression(C=self.C, max_iter=1000, random_state=42)
        self.median_ = np.median(y)
        self.max_price_ = np.max(y)
        self.model_.fit(X, (y > self.median_).astype(int))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        probs = self.model_.predict_proba(X)[:, 1]
        return probs * self.max_price_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

models = {
    'Forward_Selection': {
        'model': Pipeline([
            ('prep', preprocessor),
            ('sel', SelectKBest(score_func=f_regression)),
            ('reg', LinearRegression())
        ]),
        'params': {
            'sel__k': [20, 40, 60, 80, 100]
        }
    },
    'Logistic_Regression': {
        'model': Pipeline([
            ('prep', preprocessor_with_pca), # Uses PCA for collinearity
            ('reg', LogisticRegressorProxy())
        ]),
        'params': {
            'reg__C': [0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    'CART': {
        'model': Pipeline([('prep', preprocessor), ('reg', DecisionTreeRegressor(random_state=42))]),
        'params': {
            'reg__max_depth': [10, 15, 20],
            'reg__min_samples_split': [10, 50]
        }
    },
    'Random_Forest': {
        'model': Pipeline([('prep', preprocessor), ('reg', RandomForestRegressor(random_state=42, n_jobs=-1))]),
        'params': {
            'reg__n_estimators': [100, 200],
            'reg__max_depth': [15, 20, None]
        }
    },
    'AdaBoost': {
        'model': Pipeline([('prep', preprocessor), ('reg', AdaBoostRegressor(random_state=42))]),
        'params': {
            'reg__n_estimators': [100, 200],
            'reg__learning_rate': [0.1, 1.0]
        }
    },
    'XGBoost': {
        'model': Pipeline([('prep', preprocessor), ('reg', XGBRegressor(random_state=42, n_jobs=-1))]),
        'params': {
            'reg__n_estimators': [100, 200],
            'reg__max_depth': [5, 7, 9],
            'reg__learning_rate': [0.05, 0.1]
        }
    }
}

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
class LogisticRegressorProxy(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.model_ = None
        self.median_ = 0
        self.max_price_ = 0
        self.is_fitted_ = False

    def fit(self, X, y):
        self.model_ = LogisticRegression(C=self.C, max_iter=1000, random_state=42)
        self.median_ = np.median(y)
        self.max_price_ = np.max(y)
        self.model_.fit(X, (y > self.median_).astype(int))
        self.is_fitted_ = True
        return self

    def predict(self, X):
        probs = self.model_.predict_proba(X)[:, 1]
        return probs * self.max_price_

    def __sklearn_tags__(self):
        tags = super().__sklearn_tags__()
        return tags

best_models = {}
results = []

print("Training models & tuning 5-10 hyperparameter sets per model (Collinearity Handled)...")
for name, config in models.items():
    grid = GridSearchCV(config['model'], config['params'], cv=cv_strategy, scoring='neg_root_mean_squared_error', n_jobs=-1)
    grid.fit(X_train, y_train)

    best_models[name] = grid.best_estimator_
    val_preds = grid.predict(X_val)
    val_rmse = np.sqrt(mean_squared_error(y_val, val_preds))

    y_val_bin = (y_val > y_median).astype(int)
    val_auc = roc_auc_score(y_val_bin, val_preds)

    results.append({
        'Model': name,
        'CV_RMSE': -grid.best_score_,
        'Validation_RMSE': val_rmse,
        'Validation_AUC': val_auc
    })
    print(f"✔ {name} Completed. Valid RMSE: {val_rmse:,.0f} | AUC: {val_auc:.4f}")

results_df = pd.DataFrame(results).sort_values('Validation_RMSE')
best_name = results_df.iloc[0]['Model']
final_model = best_models[best_name]
val_preds = final_model.predict(X_val)



# ### Model Performance Assessment

# In[ ]:


fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(16, 6))

sns.barplot(data=results_df, x='Model', y='Validation_RMSE', palette="viridis", ax=ax1)
setup_mckinsey_style_axis(ax1, "Benchmark: Root Mean Squared Error", "Model Architecture", "Validation RMSE ($)")
ax1.tick_params(axis='x', rotation=45)
for i, v in enumerate(results_df['Validation_RMSE']):
    ax1.text(i, v + 100, f'${v:,.0f}', color='black', ha='center', fontweight='bold')

sns.barplot(data=results_df.sort_values('Validation_AUC', ascending=False), x='Model', y='Validation_AUC', palette="magma", ax=ax2)
setup_mckinsey_style_axis(ax2, "Classification Benchmark: Validation AUC", "Model Architecture", "Area Under ROC Curve")
ax2.tick_params(axis='x', rotation=45)
ax2.set_ylim(0, 1.05)
for i, v in enumerate(results_df.sort_values('Validation_AUC', ascending=False)['Validation_AUC']):
    ax2.text(i, v + 0.02, f'{v:.3f}', color='black', ha='center', fontweight='bold')

plt.tight_layout()
plt.show()


# ## 4. Prescriptive Strategy: Profit Maximization Offer Pricing
# 
# We sweep an algorithm across scalar variable `alpha` (Offer Price Scalar). The strategy finds the exact fraction of the Predicted Price to bid to optimize dealer profits while maintaining win probability.

# In[ ]:


def compute_profit(actual, offer):
    actual = np.array(actual)
    offer = np.array(offer)
    profit = np.zeros_like(actual)
    accepted = offer >= 0.85 * actual
    profit[accepted] = actual[accepted] - offer[accepted] - 75
    return profit.sum(), accepted.sum()

alphas = np.linspace(0.80, 0.95, 100)
profits = []
win_rates = []

for a in alphas:
    prof, wins = compute_profit(y_val, val_preds * a)
    profits.append(prof)
    win_rates.append(wins / len(y_val))

opt_idx = np.argmax(profits)
best_alpha = alphas[opt_idx]
best_profit = profits[opt_idx]
estimated_total_test_profit = (best_profit / len(y_val)) * len(test_submit)

fig, ax1 = plt.subplots(figsize=(12, 7))

color = mc_kinsey_blue
ax1.set_xlabel('Offer Price Scalar (Alpha)', fontweight='bold')
ax1.set_ylabel('Total Fleet Profit ($)', color=color, fontweight='bold')
ax1.plot(alphas, profits, color=color, linewidth=3, label='Validation Profit')
ax1.tick_params(axis='y', labelcolor=color)
ax1.axvline(best_alpha, color=accent_orange, linestyle='--', linewidth=2, 
            label=f'Optimal Bid Scalar: {best_alpha:.3f}\nMax Profit (Validation set): ${best_profit:,.0f}')
ax1.grid(True, linestyle="--", alpha=0.5)

ax2 = ax1.twinx()  
color_wr = bcg_green
ax2.set_ylabel('Acquisition Win Rate (%)', color=color_wr, fontweight='bold')  
ax2.plot(alphas, [wr * 100 for wr in win_rates], color=color_wr, linewidth=2, linestyle=':', label='Win Rate')
ax2.tick_params(axis='y', labelcolor=color_wr)

fig.legend(loc="upper left", bbox_to_anchor=(0.15,0.85))
setup_mckinsey_style_axis(ax1, "Optimization Curve: Fleet Profitability vs Bid Aggressiveness", "Offer Price Scalar (Alpha)", "Total Fleet Profit ($)")
plt.title("Optimization Curve: Fleet Profitability vs Bid Aggressiveness", loc="left", pad=20, fontweight="bold", fontsize=14)
plt.show()

print(f"\n--- OPTIMIZATION RESULTS ---")
print(f"Strategic Bid Multiplier: {best_alpha:.4f}")
print(f"Expected Validation Set Profit: ${best_profit:,.2f}  ({win_rates[opt_idx]*100:.1f}% Win Rate)")
print(f"Scaled profit estimate (validation to template size): ${estimated_total_test_profit:,.2f} -> Target: $608,680.76")


# ## 5. Execution & Deployment Strategy
# 
# Refit the selected model on all labeled training rows, then write `Cars_HW_template_predictions.xlsx`.

# In[ ]:


print(f"Refitting {best_name} on all labeled training rows...")
final_model.fit(X_full, y_full)

X_test = test_submit.drop(columns=drop_cols, errors='ignore')
test_preds = final_model.predict(X_test)
test_offers = test_preds * best_alpha

template['BestGuessAtPrice'] = test_preds
template['OfferPrice'] = test_offers
template.to_excel('Cars_HW_template_predictions.xlsx', index=False)
print(f"\nExecution complete. \nActionable predictions successfully generated for {len(test_preds):,} vehicles and injected into 'Cars_HW_template_predictions.xlsx'.")

