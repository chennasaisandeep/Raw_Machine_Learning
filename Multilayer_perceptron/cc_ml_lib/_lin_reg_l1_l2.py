"""
    linear regression with both l1 and l2
"""

import numpy as np

class LinearRegressionWithL1L2:
    def __init__(self, alpha=1.0, l1_ratio=0.5, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.l1_ratio = l1_ratio
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        # Add intercept column to X if necessary
        if self.fit_intercept:
            X = np.c_[X, np.ones(X.shape[0])]
            
        # Initialize coefficients and intercept
        coef = np.zeros(X.shape[1])
        intercept = 0
        
        # Define loss function and gradient
        def loss(X, y, coef, intercept):
            y_pred = X @ coef + intercept
            mse = np.mean((y - y_pred) ** 2)
            l1_norm = np.sum(np.abs(coef))
            l2_norm = np.sum(coef ** 2)
            return mse + self.alpha * (self.l1_ratio * l1_norm + (1 - self.l1_ratio) * l2_norm)
        
        def grad(X, y, coef, intercept):
            y_pred = X @ coef + intercept
            resid = y_pred - y
            grad_coef = 2 * X.T @ resid / X.shape[0] + self.alpha * (self.l1_ratio * np.sign(coef) + (1 - self.l1_ratio) * 2 * coef)
            grad_intercept = 2 * np.mean(resid)
            return grad_coef, grad_intercept
        
        # Run gradient descent
        for i in range(self.max_iter):
            old_coef = coef.copy()
            old_intercept = intercept
            grad_coef, grad_intercept = grad(X, y, coef, intercept)
            coef -= grad_coef * self.tol
            intercept -= grad_intercept * self.tol
            if np.max(np.abs(coef - old_coef)) < self.tol and np.abs(intercept - old_intercept) < self.tol:
                break
        
        # Set coefficients and intercept
        self.coef_ = coef
        self.intercept_ = intercept
    
    def predict(self, X):
        # Add intercept column to X if necessary
        if self.fit_intercept:
            X = np.c_[X, np.ones(X.shape[0])]
        
        # Compute predictions
        y_pred = X @ self.coef_ + self.intercept_
        return y_pred
