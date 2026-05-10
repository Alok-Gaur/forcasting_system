import numpy as np
from sklearn.metrics import root_mean_squared_error, mean_absolute_error
def evaluate_model(y_true, y_pred, model_name):

    rmse = root_mean_squared_error(y_true, y_pred)
    mae = mean_absolute_error(y_true, y_pred)

    mape = np.mean(
        np.abs((y_true - y_pred) / y_true)
    ) * 100

    return {
        'Model': model_name,
        'RMSE': rmse,
        'MAE': mae,
        'MAPE': mape
    }

results = []