from keras import backend as K
import keras.metrics as metrics
import numpy as np

def mean_squared_error(y_true, y_pred):
    return K.mean(K.square(y_pred - y_true))


def root_mean_square_error(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5


def rmse(y_true, y_pred):
    return mean_squared_error(y_true, y_pred) ** 0.5

# aliases
mse = MSE = mean_squared_error

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


def pickup_rmse(y_true, y_pred):
    pickup_y = y_true[:, 0, :, :]
    pickup_pred_y = y_pred[:, 0, :, :]
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

def test_mape(y_true, y_pred, thres=10.):
    batch_size, dims, h, w = y_true.shape

    pickup_y = np.reshape(y_true[:, 0, :, :], (batch_size, -1))
    pickup_pred_y = np.reshape(y_pred[:, 0, :, :], (batch_size, -1))

    dropoff_y = np.reshape(y_true[:, 1, :, :], (batch_size, -1))
    dropoff_pred_y = np.reshape(y_pred[:, 1, :, :], (batch_size, -1))

    pickup_mask = pickup_y > thres
    if np.sum(pickup_mask) != 0:
        avg_pickup_mape = np.mean(np.abs(pickup_y[pickup_mask] - pickup_pred_y[pickup_mask]).astype('float64') / pickup_y[pickup_mask])

    dropoff_mask = dropoff_y > thres
    if np.sum(dropoff_mask) != 0:
        avg_dropoff_mape = np.mean(np.abs(dropoff_y[dropoff_mask] - dropoff_pred_y[dropoff_mask]).astype('float64') / dropoff_y[dropoff_mask])

    return avg_pickup_mape, avg_dropoff_mape