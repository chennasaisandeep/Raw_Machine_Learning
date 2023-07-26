"""
    linear regression with l1
"""

import numpy as np

class LinearRegressionWithL1:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        n_samples, n_features = X.shape
        self.coef_ = np.zeros(n_features)
        self.intercept_ = 0.0
        
        # Add a column of ones for the intercept term
        if self.fit_intercept:
            X = np.hstack((np.ones((n_samples, 1)), X))
        
        # Define the soft-threshold function for L1 regularization
        soft_threshold = lambda x, threshold: np.maximum(0, np.abs(x) - threshold) * np.sign(x)
        
        # Run coordinate descent algorithm
        for i in range(self.max_iter):
            # Iterate over each feature and update its coefficient
            for j in range(n_features + 1):
                # Compute the partial residual
                if j == 0:
                    r_j = y - np.dot(X, self.coef_)
                else:
                    r_j = y - np.dot(X, self.coef_) + self.coef_[j-1] * X[:,j-1]
                
                # Compute the correlation between feature j and the partial residual
                corr = np.dot(X[:,j], r_j)
                
                # Update the coefficient using soft-thresholding
                if j == 0:
                    self.intercept_ = corr
                else:
                    self.coef_[j-1] = soft_threshold(corr, self.alpha) / (X[:,j-1]**2).sum()
            
            # Check for convergence
            if np.abs(self.coef_ - prev_coef).max() < self.tol:
                break
            
            prev_coef = self.coef_.copy()
    
    def predict(self, X):
        n_samples = X.shape[0]
        
        # Add a column of ones for the intercept term
        if self.fit_intercept:
            X = np.hstack((np.ones((n_samples, 1)), X))
        
        return np.dot(X, np.hstack((self.intercept_, self.coef_)))

