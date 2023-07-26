import numpy as np

class StandardScaler:
    def __init__(self):
        self.mean = None
        self.std = None
        
    def fit(self, X):
        self.mean = np.mean(X, axis=0)
        self.std = np.std(X, axis=0)
            
    def transform(self, X):
        X_transformed = (X - self.mean) / self.std
        return X_transformed