import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.base import BaseEstimator, RegressorMixin
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV

class LogisticRegressorProxy(BaseEstimator, RegressorMixin):
    def __init__(self, C=1.0):
        self.C = C
        self.model = LogisticRegression(C=C, max_iter=1000, random_state=42)
        
    def fit(self, X, y):
        self.median = np.median(y)
        self.max_price = np.max(y)
        y_bin = (y > self.median).astype(int)
        self.model.C = self.C
        self.model.fit(X, y_bin)
        return self
        
    def predict(self, X):
        probs = self.model.predict_proba(X)[:, 1]
        return probs * self.max_price

X = np.random.rand(100, 5)
y = np.random.rand(100) * 10000

pipe = Pipeline([('reg', LogisticRegressorProxy())])
grid = GridSearchCV(pipe, {'reg__C': [0.1, 1.0]}, cv=2, error_score='raise')
grid.fit(X, y)
print("Success!")
