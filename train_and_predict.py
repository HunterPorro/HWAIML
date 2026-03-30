import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, KFold, GridSearchCV
from sklearn.pipeline import Pipeline
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.feature_selection import SelectKBest, f_regression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, AdaBoostRegressor
from xgboost import XGBRegressor
from sklearn.metrics import mean_squared_error, roc_auc_score

def compute_profit(actual, offer):
    actual = np.array(actual)
    offer = np.array(offer)
    profit = np.zeros_like(actual)
    accepted = offer >= 0.85 * actual
    profit[accepted] = actual[accepted] - offer[accepted] - 75
    return profit.sum()

def main():
    print("Loading data...")
    df = pd.read_csv('Cars_HW_data.csv')
    template = pd.read_excel('Cars_HW_template.xlsx')
    
    # Split into labeled and unlabeled based on Price column
    # Actually, the template has ID which corresponds to the unlabeled data
    unlabeled_ids = set(template['ID'])
    
    train_full = df[~df['ID'].isin(unlabeled_ids)].copy()
    test_submit = df[df['ID'].isin(unlabeled_ids)].copy()
    
    # Sort test_submit to match template
    test_submit.set_index('ID', inplace=True)
    test_submit = test_submit.reindex(template['ID']).reset_index()
    
    # Features and Target
    drop_cols = ['ID', 'Price']
    if 'OfferPrice' in train_full.columns:
        drop_cols.append('OfferPrice')
    if 'BestGuessAtPrice' in train_full.columns:
        drop_cols.append('BestGuessAtPrice')
        
    X_full = train_full.drop(columns=drop_cols)
    y_full = train_full['Price']
    
    # Preprocessing
    num_cols = X_full.select_dtypes(include=['int64', 'float64']).columns
    cat_cols = X_full.select_dtypes(include=['object', 'bool']).columns
    
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
    
    # Partition 80/20
    X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
    
    print(f"Train size: {X_train.shape[0]}, Validation size: {X_val.shape[0]}")
    
    # Define models and their param grids
    models = {
        'ForwardSelection_Proxy': {
            'model': Pipeline([
                ('prep', preprocessor),
                ('sel', SelectKBest(score_func=f_regression)),
                ('reg', LinearRegression())
            ]),
            'params': {
                'sel__k': [5, 10, 20, 50, 100, 150, 'all']
            }
        },
        'Ridge_Regression': {
            'model': Pipeline([('prep', preprocessor), ('reg', Ridge())]),
            'params': {'reg__alpha': [0.1, 1.0, 10.0, 100.0, 1000.0]}
        },
        'CART': {
            'model': Pipeline([('prep', preprocessor), ('reg', DecisionTreeRegressor(random_state=42))]),
            'params': {'reg__max_depth': [5, 10, 15, None], 'reg__min_samples_split': [2, 10, 50]}
        },
        'RandomForest': {
            'model': Pipeline([('prep', preprocessor), ('reg', RandomForestRegressor(random_state=42))]),
            'params': {'reg__n_estimators': [50, 100], 'reg__max_depth': [10, 20, None], 'reg__min_samples_leaf': [1, 5]}
        },
        'AdaBoost': {
            'model': Pipeline([('prep', preprocessor), ('reg', AdaBoostRegressor(random_state=42))]),
            'params': {'reg__n_estimators': [50, 100, 200], 'reg__learning_rate': [0.01, 0.1, 1.0]}
        },
        'XGBoost': {
            'model': Pipeline([('prep', preprocessor), ('reg', XGBRegressor(random_state=42, n_jobs=-1))]),
            'params': {'reg__n_estimators': [100, 200], 'reg__max_depth': [3, 6, 9], 'reg__learning_rate': [0.01, 0.1, 0.3]}
        }
    }
    
    best_models = {}
    cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
    
    y_median = y_full.median()
    
    # For speed, let's take a sample of X_train if it's too slow, but data is < 25k so it should be fast enough.
    
    for name, config in models.items():
        print(f"\n--- Training {name} ---")
        grid = GridSearchCV(config['model'], config['params'], cv=cv_strategy, scoring='neg_root_mean_squared_error', n_jobs=-1)
        grid.fit(X_train, y_train)
        
        best_models[name] = grid.best_estimator_
        print(f"Best params: {grid.best_params_}")
        print(f"Best CV RMSE: {-grid.best_score_:.2f}")
        
        preds = grid.predict(X_val)
        val_rmse = np.sqrt(mean_squared_error(y_val, preds))
        
        y_val_bin = (y_val > y_median).astype(int)
        try:
            val_auc = roc_auc_score(y_val_bin, preds)
        except Exception as e:
            val_auc = np.nan
            
        print(f"Validation RMSE: {val_rmse:.2f}")
        print(f"Validation AUC (predicting Price > {y_median}): {val_auc:.4f}")
        
    print("\nDetermining optimal purchasing strategy...")
    
    # Choose best model overall by Validation RMSE
    best_name = min(best_models.keys(), key=lambda k: mean_squared_error(y_val, best_models[k].predict(X_val)))
    print(f"Best Model based on Validation RMSE: {best_name}")
    
    final_model = best_models[best_name]
    val_preds = final_model.predict(X_val)
    
    # Optimize scale factor to maximize profit
    best_alpha = 0.85
    best_profit = -np.inf
    # Search from 0.80 to 1.10
    alphas = np.linspace(0.80, 1.10, 301)
    
    for alpha in alphas:
        offers = val_preds * alpha
        prof = compute_profit(y_val, offers)
        if prof > best_profit:
            best_profit = prof
            best_alpha = alpha
            
    print(f"Optimal Bid Coefficient: {best_alpha:.4f}")
    print(f"Validation Set Total Profit: ${best_profit:,.2f}  ({len(y_val)} cars)")
    
    print("\nRefitting on all labeled training rows for submission predictions...")
    final_model.fit(X_full, y_full)
    
    # Predict on test_submit
    X_test = test_submit.drop(columns=drop_cols, errors='ignore')
    test_preds = final_model.predict(X_test)
    
    test_offers = test_preds * best_alpha
    
    # Save to template
    template['BestGuessAtPrice'] = test_preds
    template['OfferPrice'] = test_offers
    template.to_excel('Cars_HW_template_predictions.xlsx', index=False)
    print("\nDone! Saved to 'Cars_HW_template_predictions.xlsx'")

if __name__ == '__main__':
    main()
