"""
    implementation of logistic regression
"""

import numpy as np

class LogisticRegression:
    def __init__(self, fit_intercept=True, max_iter=100, tol=1e-4):
        self.fit_intercept = fit_intercept
        self.coef_ = None
        self.intercept_ = None
        self.max_iter = max_iter
        self.tol = tol
        
    def train(self, X, y):
        # add a column of ones to X if we want to fit an intercept
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # initialize the coefficients to zeros
        self.coef_ = np.zeros(X.shape[1])
        
        for i in range(self.max_iter):
            # calculate the predicted probabilities
            p = self.predict_proba(X)
            
            # calculate the gradient of the log-likelihood
            grad = np.dot(X.T, (p - y)) / X.shape[0]
            
            # update the coefficients
            self.coef_ -= grad
            
            # check if the change in the coefficients is smaller than the tolerance
            if np.abs(grad).max() < self.tol:
                break
    
    def predict_proba(self, X):
        # add a column of ones to X if we want to fit an intercept
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # calculate the linear combination of the inputs and the coefficients
        z = np.dot(X, self.coef_)
        
        # calculate the logistic function
        p = 1.0 / (1.0 + np.exp(-z))
        
        return p
    
    def predict(self, X, threshold=0.5):
        # add a column of ones to X if we want to fit an intercept
        if self.fit_intercept:
            X = np.hstack((np.ones((X.shape[0], 1)), X))
        
        # calculate the predicted probabilities
        p = self.predict_proba(X)
        
        # calculate the predicted labels
        y_pred = (p >= threshold).astype(int)
        
        return y_pred

