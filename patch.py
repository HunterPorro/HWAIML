import json

with open('Diamonds_Model_Comparison.ipynb', 'r') as f:
    d = json.load(f)

for cell in d['cells']:
    if cell['cell_type'] == 'code':
        source = "".join(cell['source'])
        if 'models = {' in source:
            proxy_code = """
X_train, X_val, y_train, y_val = train_test_split(X_full, y_full, test_size=0.2, random_state=42)
y_median = y_train.median()

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

from sklearn.decomposition import PCA
# Add PCA to handle collinearity among the many encoded categorical features / numerical features
# PCA(n_components=0.95) retains 95% of variance while removing collinear/redundant components.

"""
            new_source = source.replace('from sklearn.decomposition import PCA\n# Add PCA to handle collinearity among the many encoded categorical features / numerical features\n# PCA(n_components=0.95) retains 95% of variance while removing collinear/redundant components.\n', proxy_code)
            cell['source'] = [line + '\n' for line in new_source.split('\n')]
            
with open('Diamonds_Model_Comparison.ipynb', 'w') as f:
    json.dump(d, f)
