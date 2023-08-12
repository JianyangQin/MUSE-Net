# from __future__ import print_function
import numpy as np
import argparse

parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
parser.add_argument('--dataset', type=str, default='taxi', help='bike, taxi')
parser.add_argument('--device', type=str, default='2')
args = parser.parse_args()
print(args)

NO = 4
# for reproduction
seed = 1
for i in range(NO):
    seed = seed * 10 + 7
seed = seed * 10 + 7
np.random.seed(seed)
# from ipdb import set_trace
# set_trace()

# for GPU in Lab
device = args.device

import os

os.environ["CUDA_VISIBLE_DEVICES"] = device

def normalization(x):
    x_max, x_min = np.max(x), np.min(x)
    x = (x - x_min) / (x_max - x_min)
    return x

def normalization_mm(x, max, min):
    x_max, x_min = max, min
    x = (x - x_min) / (x_max - x_min)
    return x

def mean_cpt(x, num):
    avg = x[:, :2, :, :]
    for i in range(1,num):
        tmp = x[:, i*2:(i+1)*2, :, :]
        avg += tmp
    avg = avg / num
    return avg

# def pickup_rmse(y_true, y_pred, threshold=20):
#     pickup_y = y_true[:, 0, 0]
#     pickup_pred_y = y_pred[:, 0, 0]
#     mask = pickup_y > threshold
#     # avg_pickup_rmse = K.sqrt(metrics.mean_squared_error(pickup_y, pickup_pred_y))
#     avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[mask] - pickup_pred_y[mask])))
#     return avg_pickup_rmse
#
# def pickup_mae(y_true, y_pred, threshold=5):
#     pickup_y = y_true[:, 0, 0]
#     pickup_pred_y = y_pred[:, 0, 0]
#     mask = pickup_y > threshold
#     avg_pickup_mae = np.mean(np.abs(pickup_pred_y[mask] - pickup_y[mask]))
#     return avg_pickup_mae
#
# def pickup_mape(y_true, y_pred, threshold=3):
#     pickup_y = y_true[:, 0, 0]
#     pickup_pred_y = y_pred[:, 0, 0]
#     mask = pickup_y > threshold
#     if np.sum(mask) != 0:
#         mape = np.mean(np.abs(pickup_y[mask] - pickup_pred_y[mask]).astype('float64') / pickup_y[mask])
#     return mape
#
# def dropoff_rmse(y_true, y_pred, threshold=5):
#     dropoff_y = y_true[:, 0, 1]
#     dropoff_pred_y = y_pred[:, 0, 1]
#     mask = dropoff_y > threshold
#     # avg_dropoff_rmse = K.sqrt(metrics.mean_squared_error(dropoff_y, dropoff_pred_y))
#     avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[mask] - dropoff_pred_y[mask])))
#     return avg_dropoff_rmse
#
# def dropoff_mae(y_true, y_pred, threshold=5):
#     dropoff_y = y_true[:, 0, 1]
#     dropoff_pred_y = y_pred[:, 0, 1]
#     mask = dropoff_y > threshold
#     avg_dropoff_mae = np.mean(np.abs(dropoff_pred_y[mask] - dropoff_y[mask]))
#     return avg_dropoff_mae
#
# def dropoff_mape(y_true, y_pred, threshold=3):
#     dropoff_y = y_true[:, 0, 1]
#     dropoff_pred_y = y_pred[:, 0, 1]
#     mask = dropoff_y > threshold
#     if np.sum(mask) != 0:
#         mape = np.mean(np.abs(dropoff_y[mask] - dropoff_pred_y[mask]).astype('float64') / dropoff_y[mask])
#     return mape

def pickup_rmse(y_true, y_pred, threshold=10):
    pickup_y = y_true[:, 0, 0]
    pickup_pred_y = y_pred[:, 0, 0]
    mask = pickup_y > threshold
    # avg_pickup_rmse = K.sqrt(metrics.mean_squared_error(pickup_y, pickup_pred_y))
    avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y[mask] - pickup_pred_y[mask])))
    return avg_pickup_rmse

def pickup_mae(y_true, y_pred, threshold=10):
    pickup_y = y_true[:, 0, 0]
    pickup_pred_y = y_pred[:, 0, 0]
    mask = pickup_y > threshold
    avg_pickup_mae = np.mean(np.abs(pickup_pred_y[mask] - pickup_y[mask]))
    return avg_pickup_mae

def pickup_mape(y_true, y_pred, threshold=10):
    pickup_y = y_true[:, 0, 0]
    pickup_pred_y = y_pred[:, 0, 0]
    mask = pickup_y > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(pickup_y[mask] - pickup_pred_y[mask]).astype('float64') / pickup_y[mask])
    return mape

def dropoff_rmse(y_true, y_pred, threshold=10):
    dropoff_y = y_true[:, 0, 1]
    dropoff_pred_y = y_pred[:, 0, 1]
    mask = dropoff_y > threshold
    # avg_dropoff_rmse = K.sqrt(metrics.mean_squared_error(dropoff_y, dropoff_pred_y))
    avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y[mask] - dropoff_pred_y[mask])))
    return avg_dropoff_rmse

def dropoff_mae(y_true, y_pred, threshold=10):
    dropoff_y = y_true[:, 0, 1]
    dropoff_pred_y = y_pred[:, 0, 1]
    mask = dropoff_y > threshold
    avg_dropoff_mae = np.mean(np.abs(dropoff_pred_y[mask] - dropoff_y[mask]))
    return avg_dropoff_mae

def dropoff_mape(y_true, y_pred, threshold=10):
    dropoff_y = y_true[:, 0, 1]
    dropoff_pred_y = y_pred[:, 0, 1]
    mask = dropoff_y > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(dropoff_y[mask] - dropoff_pred_y[mask]).astype('float64') / dropoff_y[mask])
    return mape

class MM:
    def __init__(self, MM_max, MM_min):
        self.max = MM_max
        self.min = MM_min


def test(preds, labels, minmax):
    # test total
    outflow_rmse = pickup_rmse(labels, preds, 10)
    outflow_mae = pickup_mae(labels, preds, 10)
    outflow_mape = pickup_mape(labels, preds) * 100

    inflow_rmse = dropoff_rmse(labels, preds, 10)
    inflow_mae = dropoff_mae(labels, preds, 10)
    inflow_mape = dropoff_mape(labels, preds) * 100

    print('---------Test Score---------', end=' ')
    # print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f}'.format(
    #     outflow_rmse, outflow_mae, inflow_rmse, inflow_mae))
    print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Pickup_MAPE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f} | Dropoff_MAPE: {:04f}'.format(
            outflow_rmse, outflow_mae, outflow_mape, inflow_rmse, inflow_mae, inflow_mape))

    print('finish')


if args.dataset == 'bike':
    mm = [262, 274]
    filename = './outputs/' + args.dataset + '_0.npz'
elif args.dataset == 'taxi':
    mm = [1409, 1518]
    filename = './outputs/' + args.dataset + '_6.npz'
else:
    raise ValueError('dataset does not exist')

if os.path.exists(filename):
    data = np.load(filename)
    preds = data['preds'] * mm
    labels = data['labels'] * mm

    test(preds, labels, mm)



