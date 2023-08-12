# -*- coding:utf-8 -*-

import numpy as np
import torch


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    # print(mask.sum())
    # print(mask.shape[0]*mask.shape[1]*mask.shape[2])
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels)**2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels, null_val=np.nan):
    return torch.sqrt(masked_mse(preds=preds, labels=labels,
                                 null_val=null_val))


def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds - labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)



def masked_mae_test(y_true, y_pred, null_val=np.nan):
    mae = np.abs(np.subtract(y_true, y_pred)).astype(np.float32)
    mae = np.mean(mae)
    return mae


def masked_rmse_test(y_true, y_pred, null_val=np.nan):
    rmse = np.abs(np.subtract(y_true, y_pred)).astype(np.float32)
    rmse = np.square(rmse)
    rmse = np.sqrt(np.mean(rmse))
    return rmse


def masked_mape_test(y_true, y_pred, threshold=10.):
    mask = y_true > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(y_true[mask] - y_pred[mask]) / y_true[mask])
    return mape