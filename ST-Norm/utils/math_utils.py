
import numpy as np


def MAPE(v, v_, threshold=10):
    # mape = np.mean(np.abs(v - v_) / np.maximum(0.05, v + 1.0))
    # return mape
    greater_than_threshold = v > threshold
    v = v[greater_than_threshold]
    v_ = v_[greater_than_threshold]
    if np.sum(greater_than_threshold) != 0:
        mape = np.mean(np.abs(v_ - v) / (v))
    return mape


def RMSE(v, v_):
    return np.sqrt(np.mean((v_ - v) ** 2))


def MAE(v, v_):
    return np.mean(np.abs(v_ - v))


def evaluation(y, y_, x_stats):
    dim = len(y_.shape)

    if dim == 3:
        # single_step case
        mm = x_stats['max'] - x_stats['min']
        return np.array([MAPE((y * mm + mm) / 2., (y_ * mm + mm) / 2.), MAE(y, y_) * mm / 2, RMSE(y, y_) * mm / 2])
        # return np.array([MAPE(y, y_), MAE(y, y_) * mm / 2, RMSE(y, y_) * mm / 2])
    else:
        # multi_step case
        inflow_list, outflow_list = [], []
        # y -> [time_step, batch_size, n_route, 1]
        y = np.swapaxes(y, 0, 1)
        y_ = np.swapaxes(y_, 0, 1)
        # recursively call
        for i in range(y_.shape[0]):
            inflow_res = evaluation(y[i, :, :, 1:2], y_[i, :, :, 1:2], x_stats)
            outflow_res = evaluation(y[i, :, :, 0:1], y_[i, :, :, 0:1], x_stats)
            inflow_list.append(inflow_res)
            outflow_list.append(outflow_res)
        return np.concatenate(outflow_list, axis=-1), np.concatenate(inflow_list, axis=-1)
