"""
    linear regression with l2
"""

import numpy as np

class LinearRegressionWithL2:
    def __init__(self, alpha=1.0, fit_intercept=True, max_iter=1000, tol=1e-4):
        self.alpha = alpha
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.tol = tol
    
    def fit(self, X, y):
        # Add a column of 1s to X if fit_intercept is True
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Initialize the coefficients to 0
        self.coef_ = np.zeros(X.shape[1])
        
        # Define the L2 regularization term
        l2_term = self.alpha * np.eye(X.shape[1])
        l2_term[0, 0] = 0.0  # Don't regularize the intercept
        
        # Perform gradient descent to optimize the coefficients
        for i in range(self.max_iter):
            # Calculate the predicted values
            y_pred = X @ self.coef_
            
            # Calculate the error
            error = y_pred - y
            
            # Calculate the gradient
            gradient = X.T @ error + l2_term @ self.coef_
            
            # Update the coefficients
            self.coef_ -= gradient * self.tol
            
            # Check convergence
            if np.sum(np.abs(gradient)) < self.tol:
                break
        
        # Extract the intercept and coefficients
        self.intercept_ = self.coef_[0]
        self.coef_ = self.coef_[1:]
    
    def predict(self, X):
        # Add a column of 1s to X if fit_intercept is True
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # Predict the target values using the optimized coefficients
        y_pred = X @ np.hstack((self.intercept_, self.coef_))
        
        return y_pred
