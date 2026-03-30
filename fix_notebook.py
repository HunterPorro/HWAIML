import json

with open('Diamonds_Model_Comparison.ipynb', 'r') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'class LogisticRegressorProxy' in source:
            new_source = """
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
y_median = y_train.median()

class LogisticRegressorProxy(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0):
        self.C = C
        
    def fit(self, X, y):
        self.model_ = LogisticRegression(C=self.C, max_iter=1000, random_state=42)
        self.median_ = np.median(y)
        self.max_price_ = np.max(y)
        y_bin = (y > self.median_).astype(int)
        self.model_.fit(X, y_bin)
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
            'sel__k': [10, 20, 30, 40, 50, 60, 80]
        }
    },
    'Logistic_Regression': {
        'model': Pipeline([
            ('prep', preprocessor),
            ('reg', LogisticRegressorProxy())
        ]),
        'params': {
            'reg__C': [0.001, 0.01, 0.1, 1.0, 10.0, 100.0]
        }
    },
    'CART': {
        'model': Pipeline([('prep', preprocessor), ('reg', DecisionTreeRegressor(random_state=42))]),
        'params': {
            'reg__max_depth': [5, 10, 15],
            'reg__min_samples_split': [10, 50]
        }
    },
    'Random_Forest': {
        'model': Pipeline([('prep', preprocessor), ('reg', RandomForestRegressor(random_state=42, n_jobs=1))]),
        'params': {
            'reg__n_estimators': [50, 100],
            'reg__max_depth': [10, 20, None]
        }
    },
    'AdaBoost': {
        'model': Pipeline([('prep', preprocessor), ('reg', AdaBoostRegressor(random_state=42))]),
        'params': {
            'reg__n_estimators': [50, 100, 150],
            'reg__learning_rate': [0.01, 0.1, 1.0]
        }
    },
    'XGBoost': {
        'model': Pipeline([('prep', preprocessor), ('reg', XGBRegressor(random_state=42, n_jobs=1))]),
        'params': {
            'reg__n_estimators': [50, 100, 150],
            'reg__max_depth': [3, 5, 7]
        }
    }
}

cv_strategy = KFold(n_splits=5, shuffle=True, random_state=42)
best_models = {}
results = []

print("Training models & tuning 5-10 hyperparameter sets per model...")
for name, config in models.items():
    grid = GridSearchCV(config['model'], config['params'], cv=cv_strategy, scoring='neg_root_mean_squared_error', n_jobs=1)
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
            cell['source'] = [line + '\n' for line in new_source.split('\n')]

with open('Diamonds_Model_Comparison.ipynb', 'w') as f:
    json.dump(d, f)
