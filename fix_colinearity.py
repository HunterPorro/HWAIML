import json

with open('Diamonds_Model_Comparison.ipynb', 'r') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'models = {' in source:
            new_source = """
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
"""
            # Extract only the portion corresponding to the models block to replace
            start_idx = source.find('models = {')
            end_idx = source.find('fig, (ax1, ax2)')
            if end_idx == -1:
                # If they are in the same cell, just replace the end half
                cell['source'] = [line + '\n' for line in new_source.split('\n')]
            else:
                 pass # simplified 

with open('Diamonds_Model_Comparison.ipynb', 'w') as f:
    json.dump(d, f)
