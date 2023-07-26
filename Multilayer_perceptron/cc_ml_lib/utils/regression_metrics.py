import numpy as np

def rmse(y_actual, y_pred):
    """Calculate the Root Mean Squared Error (RMSE) between two arrays"""
    mse = np.mean((y_pred - y_actual) ** 2)
    rmse = np.sqrt(mse)
    return rmse

def r2(y_actual, y_pred):
    """Calculate the R-squared (R2) between two arrays"""
    sse = np.sum((y_actual - y_pred)**2)
    sst = np.sum((y_actual - np.mean(y_actual))**2)
    r2 = 1 - (sse / sst)
    return r2
    

def mape(y_actual, y_pred):
    mape = np.mean(np.abs((y_actual - y_pred) / y_actual)) * 100
    return mape
