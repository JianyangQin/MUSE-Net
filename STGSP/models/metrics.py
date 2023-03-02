# -*- coding:utf-8 -*-

import numpy as np
import torch

def masked_mape_np(y_true, y_pred):
    return np.abs(np.divide(np.subtract(y_pred, y_true).astype('float32'), y_true))

def masked_mse(preds, labels):
    return np.mean((preds - labels) ** 2)

def masked_rmse(preds, labels):
    return np.sqrt(np.mean((preds - labels) ** 2))

def masked_mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))

def masked_mae_test(y_true, y_pred):
    return np.mean(np.abs(y_true - y_pred))

def masked_rmse_test(y_true, y_pred):
    return np.sqrt(np.mean((y_true - y_pred) ** 2))

def masked_mape_test(y_true, y_pred, threshold=10.0):
    mask = y_true > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(y_true[mask] - y_pred[mask]).astype('float32') / y_true[mask])
    return mape * 100
