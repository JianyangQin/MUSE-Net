import numpy as np
from keras import backend as K

def pickup_rmse(y_true, y_pred):
    pickup_y = y_true[:, :, 0]
    pickup_pred_y = y_pred[:, :, 0]
    # avg_pickup_rmse = K.sqrt(metrics.mean_squared_error(pickup_y, pickup_pred_y))
    avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y - pickup_pred_y)))
    return avg_pickup_rmse

def pickup_mae(y_true, y_pred):
    pickup_y = y_true[:, :, 0]
    pickup_pred_y = y_pred[:, :, 0]
    avg_pickup_mae = np.mean(np.abs(pickup_pred_y - pickup_y))
    return avg_pickup_mae

def pickup_mape(y_true, y_pred, threshold=10):
    pickup_y = y_true[:, :, 0]
    pickup_pred_y = y_pred[:, :, 0]
    mask = pickup_y > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(pickup_y[mask] - pickup_pred_y[mask]).astype('float64') / pickup_y[mask])
    return mape

def dropoff_rmse(y_true, y_pred):
    dropoff_y = y_true[:, :, 1]
    dropoff_pred_y = y_pred[:, :, 1]
    # avg_dropoff_rmse = K.sqrt(metrics.mean_squared_error(dropoff_y, dropoff_pred_y))
    avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y - dropoff_pred_y)))
    return avg_dropoff_rmse

def dropoff_mae(y_true, y_pred):
    dropoff_y = y_true[:, :, 1]
    dropoff_pred_y = y_pred[:, :, 1]
    avg_dropoff_mae = np.mean(np.abs(dropoff_pred_y - dropoff_y))
    return avg_dropoff_mae

def dropoff_mape(y_true, y_pred, threshold=10):
    dropoff_y = y_true[:, :, 1]
    dropoff_pred_y = y_pred[:, :, 1]
    mask = dropoff_y > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(dropoff_y[mask] - dropoff_pred_y[mask]).astype('float64') / dropoff_y[mask])
    return mape

def evaluate_performance(labels, preds):
    outflow_rmse = pickup_rmse(preds, labels)
    outflow_mae = pickup_mae(preds, labels)
    outflow_mape = pickup_mape(labels, preds) * 100

    inflow_rmse = dropoff_rmse(preds, labels)
    inflow_mae = dropoff_mae(preds, labels)
    inflow_mape = dropoff_mape(labels, preds) * 100
    return outflow_rmse, outflow_mae, outflow_mape, inflow_rmse, inflow_mae, inflow_mape