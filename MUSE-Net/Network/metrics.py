import numpy as np
from keras import backend as K
import keras.metrics as metrics


def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


# aliases
mse = MSE = mean_squared_error


# rmse = RMSE = root_mean_square_error


def masked_mean_squared_error(y_true, y_pred):
    idx = (y_true > 1e-6).nonzero()
    return K.mean(K.square(y_pred[idx] - y_true[idx]))


def masked_rmse(y_true, y_pred):
    return masked_mean_squared_error(y_true, y_pred) ** 0.5


def mean_absolute_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


def mae(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true))


threshold = 0.05


def mean_absolute_percentage_error(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true) / K.maximum(K.cast(threshold, 'float32'), y_true + 1.0))


def mape(y_true, y_pred):
    return K.mean(K.abs(y_pred - y_true) / K.maximum(K.cast(threshold, 'float32'), y_true + 1.0))


# def evaluate(y_true, y_pred):
#     threshold = 0.1
#     pickup_y_true = K.reshape(y_true[:, 0, :, :], (-1,))
#     dropoff_y_true = K.reshape(y_true[:, 1, :, :], (-1))
#     pickup_y_pred = K.reshape(y_pred[:, 0, :, :], (-1))
#     dropoff_y_pred = K.reshape(y_pred[:, 1, :, :], (-1))
#     pickup_mask = K.cast(pickup_y_true > threshold, 'int32')
#     dropoff_mask = K.cast(dropoff_y_true > threshold, 'int32')
#     # pickup part
#     if K.sum(pickup_mask) != 0:
#         avg_pickup_mape = K.mean(K.abs(K.gather(pickup_y_true, pickup_mask) - K.gather(pickup_y_pred, pickup_mask)) / pickup_y_true[pickup_mask])
#         avg_pickup_rmse = K.sqrt(K.mean(K.square(pickup_y_true[pickup_mask] - pickup_y_pred[pickup_mask])))
#     # dropoff part
#     if K.sum(dropoff_mask) != 0:
#         avg_dropoff_mape = K.mean(K.abs(dropoff_y_true[dropoff_mask] - dropoff_y_pred[dropoff_mask]) / dropoff_y_true[dropoff_mask])
#         avg_dropoff_rmse = K.sqrt(K.mean(K.square(dropoff_y_true[dropoff_mask] - dropoff_y_pred[dropoff_mask])))
#
#     return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)

# def evaluate(y_true, y_pred):
#     pickup_y = y_true[:, 0, :, :]
#     dropoff_y = y_true[:, 1, :, :]
#     pickup_pred_y = y_pred[:, 0, :, :]
#     dropoff_pred_y = y_pred[:, 1, :, :]
#     # pickup part
#     avg_pickup_mape = K.mean(K.abs(pickup_y - pickup_pred_y) / K.maximum(K.cast(threshold,'float32'), pickup_y))
#     avg_pickup_rmse = K.sqrt(K.mean(K.square(pickup_y - pickup_pred_y)))
#     # dropoff part
#     avg_dropoff_mape = K.mean(K.abs(dropoff_y - dropoff_pred_y) / K.maximum(K.cast(threshold,'float32'), dropoff_y))
#     avg_dropoff_rmse = K.sqrt(K.mean(K.square(dropoff_y - dropoff_pred_y)))
#
#     return (avg_pickup_rmse, avg_pickup_mape), (avg_dropoff_rmse, avg_dropoff_mape)

def pickup_rmse(y_true, y_pred):
    pickup_y = y_true[:, 0, :, :]
    pickup_pred_y = y_pred[:, 0, :, :]
    # avg_pickup_rmse = K.sqrt(metrics.mean_squared_error(pickup_y, pickup_pred_y))
    avg_pickup_rmse = K.sqrt(K.mean(K.square(pickup_y - pickup_pred_y)))
    return avg_pickup_rmse

def pickup_mae(y_true, y_pred):
    pickup_y = y_true[:, 0, :, :]
    pickup_pred_y = y_pred[:, 0, :, :]
    avg_pickup_mae = K.mean(K.abs(pickup_pred_y - pickup_y))
    return avg_pickup_mae

def pickup_mape(y_true, y_pred):
    pickup_y = y_true[:, 0, :, :]
    pickup_pred_y = y_pred[:, 0, :, :]
    avg_pickup_mape = K.mean(K.abs(pickup_y - pickup_pred_y) / K.maximum(K.cast(threshold, 'float32'), pickup_y))
    return avg_pickup_mape


def dropoff_rmse(y_true, y_pred):
    dropoff_y = y_true[:, 1, :, :]
    dropoff_pred_y = y_pred[:, 1, :, :]
    # avg_dropoff_rmse = K.sqrt(metrics.mean_squared_error(dropoff_y, dropoff_pred_y))
    avg_dropoff_rmse = K.sqrt(K.mean(K.square(dropoff_y - dropoff_pred_y)))
    return avg_dropoff_rmse

def dropoff_mae(y_true, y_pred):
    dropoff_y = y_true[:, 1, :, :]
    dropoff_pred_y = y_pred[:, 1, :, :]
    avg_dropoff_mae = K.mean(K.abs(dropoff_pred_y - dropoff_y))
    return avg_dropoff_mae

def dropoff_mape(y_true, y_pred):
    dropoff_y = y_true[:, 1, :, :]
    dropoff_pred_y = y_pred[:, 1, :, :]
    avg_dropoff_mape = K.mean(K.abs(dropoff_y - dropoff_pred_y) / K.maximum(K.cast(threshold, 'float32'), dropoff_y))
    return avg_dropoff_mape

# def metrics(y_true, y_pred, max, min):
#     outflow_rmse = pickup_rmse(y_true, y_pred) * (max - min) / 2
#     outflow_mae = pickup_mae(y_true, y_pred) * (max - min) / 2
#     outflow_mape = pickup_mape((y_true * (max - min) + (max + min)) / 2., (y_pred * (max - min) + (max + min) ) / 2.) * 100
#
#     inflow_rmse = dropoff_rmse(y_true, y_pred) * (max - min) / 2
#     inflow_mae = dropoff_mae(y_true, y_pred) * (max - min) / 2
#     inflow_mape = dropoff_mape((y_true * (max - min) + (max + min)) / 2., (y_pred * (max - min) + (max + min)) / 2.) * 100
#     return [outflow_mae, outflow_rmse, outflow_mape, inflow_mae, inflow_rmse, inflow_mape]