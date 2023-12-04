# from __future__ import print_function
import numpy as np
import argparse
import pandas as pd
import h5py
import matplotlib.pyplot as plt
from Network.metrics import test_mape

parser = argparse.ArgumentParser(description='Spatial-Temporal Dynamic Network')
parser.add_argument('--dataset', type=str, default='TaxiBJ', help='nycbike, nyctaxi or taxibj')
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
import setproctitle

setproctitle.setproctitle('Comprison Start! @ Love Little GuoGuo')  # from V1707

from keras import backend as K

K.set_image_data_format('channels_first')

# hyperparameters
epoch = 350  # number of epoch at training stage
batch_size = 8  # batch size
lr = 0.0002  # learning rate

H, W, channel = 10, 20, 2  # grid size

T = 24 * 2  # number of time intervals in one day

len_closeness = 3  # length of closeness dependent sequence
len_period = 4  # length of peroid dependent sequence
len_trend = 4  # length of trend dependent sequence

T_closeness, T_period, T_trend = 1, T, T * 7

# last 7 days for testing data
days_test = 20
len_test = T * days_test

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

def pickup_rmse(y_true, y_pred):
    pickup_y = y_true[:, 0, :, :]
    pickup_pred_y = y_pred[:, 0, :, :]
    # avg_pickup_rmse = K.sqrt(metrics.mean_squared_error(pickup_y, pickup_pred_y))
    avg_pickup_rmse = np.sqrt(np.mean(np.square(pickup_y - pickup_pred_y)))
    return avg_pickup_rmse

def pickup_mae(y_true, y_pred):
    pickup_y = y_true[:, 0, :, :]
    pickup_pred_y = y_pred[:, 0, :, :]
    avg_pickup_mae = np.mean(np.abs(pickup_pred_y - pickup_y))
    return avg_pickup_mae

def pickup_mape(y_true, y_pred, threshold=10):
    pickup_y = y_true[:, 0, :, :]
    pickup_pred_y = y_pred[:, 0, :, :]
    mask = pickup_y > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(pickup_y[mask] - pickup_pred_y[mask]).astype('float64') / pickup_y[mask])
    return mape

def dropoff_rmse(y_true, y_pred):
    dropoff_y = y_true[:, 1, :, :]
    dropoff_pred_y = y_pred[:, 1, :, :]
    # avg_dropoff_rmse = K.sqrt(metrics.mean_squared_error(dropoff_y, dropoff_pred_y))
    avg_dropoff_rmse = np.sqrt(np.mean(np.square(dropoff_y - dropoff_pred_y)))
    return avg_dropoff_rmse

def dropoff_mae(y_true, y_pred):
    dropoff_y = y_true[:, 1, :, :]
    dropoff_pred_y = y_pred[:, 1, :, :]
    avg_dropoff_mae = np.mean(np.abs(dropoff_pred_y - dropoff_y))
    return avg_dropoff_mae

def dropoff_mape(y_true, y_pred, threshold=10):
    dropoff_y = y_true[:, 1, :, :]
    dropoff_pred_y = y_pred[:, 1, :, :]
    mask = dropoff_y > threshold
    if np.sum(mask) != 0:
        mape = np.mean(np.abs(dropoff_y[mask] - dropoff_pred_y[mask]).astype('float64') / dropoff_y[mask])
    return mape

class MM:
    def __init__(self, MM_max, MM_min):
        self.max = MM_max
        self.min = MM_min

def load_npy_data(dataset, len_test, len_closeness, len_period, len_trend, T_closeness=1, T_period=24, T_trend=24 * 7):
    all_data = np.load(dataset)
    len_total, feature, map_height, map_width = all_data.shape
    print('all_data shape: ', all_data.shape)
    mm = MM(np.max(all_data), np.min(all_data))
    print('max=', mm.max, ' min=', mm.min)

    # for time
    time = np.arange(len_total, dtype=int)
    # hour
    time_hour = time % T_period
    matrix_hour = np.zeros([len_total, T_period, map_height, map_width])
    for i in range(len_total):
        matrix_hour[i, time_hour[i], :, :] = 1
    # day
    time_day = (time // T_period) % 7
    matrix_day = np.zeros([len_total, 7, map_height, map_width])
    for i in range(len_total):
        matrix_day[i, time_day[i], :, :] = 1
    # con
    matrix_T = np.concatenate((matrix_hour, matrix_day), axis=1)

    all_data=(2.0*all_data-(mm.max+mm.min))/(mm.max-mm.min)
    print('mean=', np.mean(all_data), ' variance=', np.std(all_data))

    if len_trend > 0:
        number_of_skip_hours = T_trend * len_trend
    elif len_period > 0:
        number_of_skip_hours = T_period * len_period
    elif len_closeness > 0:
        number_of_skip_hours = T_closeness * len_closeness
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:len_total]

    if len_closeness > 0:
        X_closeness = all_data[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(len_closeness - 1):
            X_closeness = np.concatenate(
                (X_closeness, all_data[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    if len_period > 0:
        X_period = all_data[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(len_period - 1):
            X_period = np.concatenate(
                (X_period, all_data[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]), axis=1)
    if len_trend > 0:
        X_trend = all_data[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(len_trend - 1):
            X_trend = np.concatenate(
                (X_trend, all_data[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]), axis=1)

    matrix_T = matrix_T[number_of_skip_hours:]

    X_closeness_train = X_closeness[:-len_test]
    X_period_train = X_period[:-len_test]
    X_trend_train = X_trend[:-len_test]
    T_train = matrix_T[:-len_test]
    X_closeness_test = X_closeness[-len_test:]
    X_period_test = X_period[-len_test:]
    X_trend_test = X_trend[-len_test:]
    T_test = matrix_T[-len_test:]

    X_train = [X_closeness_train, X_period_train, X_trend_train]
    X_test = [X_closeness_test, X_period_test, X_trend_test]
    Y_train = Y[:-len_test]
    Y_test = Y[-len_test:]

    len_train = X_closeness_train.shape[0]
    len_test = X_closeness_test.shape[0]
    print('len_train=' + str(len_train))
    print('len_test =' + str(len_test))

    return X_train, T_train, Y_train, X_test, T_test, Y_test, mm.max - mm.min, mm.max, mm.min, len_total, len_test, number_of_skip_hours

def remove_incomplete_days(data, timestamps, T=48):
    """
    remove a certain day which has not 48 timestamps
    :param data:
    :param timestamps:
    :param T:
    :return:
    """

    days = []  # available days: some day only contain some seqs
    days_incomplete = []
    i = 0
    while i < len(timestamps):
        if int(timestamps[i][8:]) != 1:
            i += 1
        elif i + T - 1 < len(timestamps) and int(timestamps[i + T - 1][8:]) == T:
            days.append(timestamps[i][:8])
            i += T
        else:
            days_incomplete.append(timestamps[i][:8])
            i += 1
    print("incomplete days: ", days_incomplete)
    days = set(days)
    idx = []
    for i, t in enumerate(timestamps):
        if t[:8] in days:
            idx.append(i)

    data = data[idx]
    timestamps = [timestamps[i] for i in idx]
    return data, timestamps

def load_taxibj(dataset, len_test, len_closeness, len_period, len_trend, T_closeness=1, T_period=24, T_trend=24 * 7, T=48):
    f = h5py.File(dataset, 'r')
    all_data = f['data'].value
    timestamps = f['date'].value
    f.close()
    all_data, _ = remove_incomplete_days(all_data, timestamps, T)

    len_total, feature, map_height, map_width = all_data.shape
    print('all_data shape: ', all_data.shape)
    mm = MM(np.max(all_data), np.min(all_data))
    print('max=', mm.max, ' min=', mm.min)

    # for time
    time = np.arange(len_total, dtype=int)
    # hour
    time_hour = time % T_period
    matrix_hour = np.zeros([len_total, T_period, map_height, map_width])
    for i in range(len_total):
        matrix_hour[i, time_hour[i], :, :] = 1
    # day
    time_day = (time // T_period) % 7
    matrix_day = np.zeros([len_total, 7, map_height, map_width])
    for i in range(len_total):
        matrix_day[i, time_day[i], :, :] = 1
    # con
    matrix_T = np.concatenate((matrix_hour, matrix_day), axis=1)

    all_data=(2.0*all_data-(mm.max+mm.min))/(mm.max-mm.min)
    print('mean=', np.mean(all_data), ' variance=', np.std(all_data))

    if len_trend > 0:
        number_of_skip_hours = T_trend * len_trend
    elif len_period > 0:
        number_of_skip_hours = T_period * len_period
    elif len_closeness > 0:
        number_of_skip_hours = T_closeness * len_closeness
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:len_total]

    if len_closeness > 0:
        X_closeness = all_data[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(len_closeness - 1):
            X_closeness = np.concatenate(
                (X_closeness, all_data[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    if len_period > 0:
        X_period = all_data[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(len_period - 1):
            X_period = np.concatenate(
                (X_period, all_data[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]), axis=1)
    if len_trend > 0:
        X_trend = all_data[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(len_trend - 1):
            X_trend = np.concatenate(
                (X_trend, all_data[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]), axis=1)

    matrix_T = matrix_T[number_of_skip_hours:]

    X_closeness_train = X_closeness[:-len_test]
    X_period_train = X_period[:-len_test]
    X_trend_train = X_trend[:-len_test]
    T_train = matrix_T[:-len_test]
    X_closeness_test = X_closeness[-len_test:]
    X_period_test = X_period[-len_test:]
    X_trend_test = X_trend[-len_test:]
    T_test = matrix_T[-len_test:]

    X_train = [X_closeness_train, X_period_train, X_trend_train]
    X_test = [X_closeness_test, X_period_test, X_trend_test]
    Y_train = Y[:-len_test]
    Y_test = Y[-len_test:]

    len_train = X_closeness_train.shape[0]
    len_test = X_closeness_test.shape[0]
    print('len_train=' + str(len_train))
    print('len_test =' + str(len_test))

    return X_train, T_train, Y_train, X_test, T_test, Y_test, mm.max - mm.min, mm.max, mm.min, len_total, len_test, number_of_skip_hours

def get_Time(start_time, total_steps, test_steps, skip_steps):
    freq = '30min'
    freq_delta_second = 60 * 30
    t = pd.date_range(start_time, periods=total_steps, freq=freq)
    df = pd.DataFrame({"date": t})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index("date", inplace=True)
    Time = df.index
    dayofweek = np.reshape(Time.weekday, newshape=(-1, 1))
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) \
                // freq_delta_second
    timeofday = np.reshape(timeofday, newshape=(-1, 1))
    Time = np.concatenate((dayofweek, timeofday), axis=-1)
    # train/val/test
    total_Time = Time[skip_steps:total_steps]
    train = total_Time[:-test_steps]
    test = total_Time[-test_steps:]
    return train, test

def get_WeekdayPeriod(x, t):
    len = x.shape[0]
    x_weekday, x_weekend, x_mask = [], [], []
    for i in range(len):
        if t[i][0] == 5 or t[i][0] == 6:
            x_mask.append([0])
            x_weekend.append(np.expand_dims(x[i], axis=0))
        else:
            x_mask.append([1])
            x_weekday.append(np.expand_dims(x[i], axis=0))
    x_weekday = np.concatenate(x_weekday, axis=0)
    x_weekend = np.concatenate(x_weekend, axis=0)
    x_mask = np.concatenate(x_mask, axis=0)
    return x_weekday, x_weekend, x_mask

def get_PeekPeriod(x, t):
    len = x.shape[0]
    x_peak, x_nonpeak, x_mask = [], [], []
    for i in range(len):
        if (t[i][1] > 13 and t[i][1] < 19) or (t[i][1] > 33 and t[i][1] < 39):
            x_mask.append([1])
            x_peak.append(np.expand_dims(x[i], axis=0))
        else:
            x_mask.append([0])
            x_nonpeak.append(np.expand_dims(x[i], axis=0))
    x_peak = np.concatenate(x_peak, axis=0)
    x_nonpeak = np.concatenate(x_nonpeak, axis=0)
    x_mask = np.concatenate(x_mask, axis=0)
    return x_peak, x_nonpeak, x_mask

def test_period(preds, labels, t, minmax, get_period, pos_name='Weekday', neg_name='Weekend', viz=False):
    # test total
    outflow_rmse = pickup_rmse(preds, labels) * minmax / 2
    outflow_mae = pickup_mae(preds, labels) * minmax / 2
    outflow_mape = pickup_mape((labels * minmax + minmax) / 2., (preds * minmax + minmax) / 2.) * 100

    inflow_rmse = dropoff_rmse(preds, labels) * minmax / 2
    inflow_mae = dropoff_mae(preds, labels) * minmax / 2
    inflow_mape = dropoff_mape((labels * minmax + minmax) / 2., (preds * minmax + minmax) / 2.) * 100

    print('---------Test Score---------', end=' ')
    # print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f}'.format(
    #     outflow_rmse, outflow_mae, inflow_rmse, inflow_mae))
    print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Pickup_MAPE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f} | Dropoff_MAPE: {:04f}'.format(
            outflow_rmse, outflow_mae, outflow_mape, inflow_rmse, inflow_mae, inflow_mape))

    preds_pos, preds_neg, preds_mask = get_period(preds, t)
    labels_pos, labels_neg, labels_mask = get_period(labels, t)

    # test weekday
    pos_outflow_rmse = pickup_rmse(preds_pos, labels_pos) * minmax / 2
    pos_outflow_mae = pickup_mae(preds_pos, labels_pos) * minmax / 2
    pos_outflow_mape = pickup_mape((labels_pos * minmax + minmax) / 2., (preds_pos * minmax + minmax) / 2.) * 100

    pos_inflow_rmse = dropoff_rmse(preds_pos, labels_pos) * minmax / 2
    pos_inflow_mae = dropoff_mae(preds_pos, labels_pos) * minmax / 2
    pos_inflow_mape = dropoff_mape((labels_pos * minmax + minmax) / 2., (preds_pos * minmax + minmax) / 2.) * 100

    print('-----{0} Test Score-----'.format(pos_name), end=' ')
    # print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f}'.format(
    #     pos_outflow_rmse, pos_outflow_mae, pos_inflow_rmse, pos_inflow_mae))
    print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Pickup_MAPE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f} | Dropoff_MAPE: {:04f}'.format(
        pos_outflow_rmse, pos_outflow_mae, pos_outflow_mape, pos_inflow_rmse, pos_inflow_mae, pos_inflow_mape))

    # test weekend
    neg_outflow_rmse = pickup_rmse(preds_neg, labels_neg) * minmax / 2
    neg_outflow_mae = pickup_mae(preds_neg, labels_neg) * minmax / 2
    neg_outflow_mape = pickup_mape((labels_neg * minmax + minmax) / 2., (preds_neg * minmax + minmax) / 2.) * 100

    neg_inflow_rmse = dropoff_rmse(preds_neg, labels_neg) * minmax / 2
    neg_inflow_mae = dropoff_mae(preds_neg, labels_neg) * minmax / 2
    neg_inflow_mape = dropoff_mape((labels_neg * minmax + minmax) / 2., (preds_neg * minmax + minmax) / 2.) * 100

    print('-----{0} Test Score-----'.format(neg_name), end=' ')
    # print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f}'.format(
    #     neg_outflow_rmse, neg_outflow_mae, neg_inflow_rmse, neg_inflow_mae))
    print(
        'Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Pickup_MAPE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f} | Dropoff_MAPE: {:04f}'.format(
            neg_outflow_rmse, neg_outflow_mae, neg_outflow_mape, neg_inflow_rmse, neg_inflow_mae, neg_inflow_mape))

    if viz is True:
        timestep = labels.shape[0]
        x = np.arange(0, timestep, 1)

        H, W = 2, 5

        pathname = './' + args.dataset + '_flow/' + args.dataset + '_'
        for i in range(H):
            for j in range(W):
                k = i * W + j

                # inflow
                fig_in = plt.figure(figsize=(18, 6))
                plt.rc('font', family='Times New Roman')
                ax_in = fig_in.add_axes([0.13, 0.30, 0.82, 0.50])
                pred_inflow = preds[:, 0, i, j]
                label_inflow = labels[:, 0, i, j]
                ax_in.tick_params(labelsize=56)
                for xx in range(len(x)):
                    if preds_mask[xx] == 1:
                        plt.axvline(x=xx, ls="-", c="palegreen")  # 添加垂直直线
                ax_in.plot(x, pred_inflow, color="red", linewidth=2, label='prediction')
                ax_in.plot(x, label_inflow, color="blue", linewidth=2, label='ground-truth')
                # ax_in.legend(fontsize=56, loc='upper right') #自动检测要在图例中显示的元素，并且显示
                ax_in.set_xlabel('tiemslot', fontsize=64)
                ax_in.set_ylabel('flow', fontsize=64)
                plt.title('Inflow', fontsize=72)
                # plt.savefig(pathname + str(k) + '_inflow.png', bbox_inches='tight')
                plt.show()

                # outflow
                fig_out = plt.figure(figsize=(18, 6))
                plt.rc('font', family='Times New Roman')
                ax_out = fig_out.add_axes([0.13, 0.30, 0.82, 0.50])
                pred_inflow = preds[:, 0, i, j]
                label_inflow = labels[:, 0, i, j]
                plt.tick_params(labelsize=56)
                for xx in range(len(x)):
                    if preds_mask[xx] == 1:
                        plt.axvline(x=xx, ls="-", c="palegreen")  # 添加垂直直线
                ax_out.plot(x, pred_inflow, color="red", linewidth=2, label='prediction')
                ax_out.plot(x, label_inflow, color="blue", linewidth=2, label='ground-truth')
                # ax_out.legend(fontsize=48, loc='upper right') #自动检测要在图例中显示的元素，并且显示
                plt.xlabel('tiemslot', fontsize=64)
                plt.ylabel('flow', fontsize=64)
                plt.title('Outflow', fontsize=72)
                # plt.savefig(pathname + str(k) + '_outflow.png', bbox_inches='tight')
                plt.show()

        print('finish')


def evaluate_period(model, x, y, t, minmax, get_period, pos_name='Weekday', neg_name='Weekend'):
    x_pos, x_neg, x_mask = get_period(x, t)
    y_pos, y_neg, y_mask = get_period(y, t)

    pos_score = model.evaluate(x_pos, y_pos, batch_size=batch_size, verbose=0)
    pos_pred = model.predict(x_pos, batch_size=batch_size)

    pos_outflow_rmse = pos_score[1] * MinMax / 2
    pos_outflow_mae = pos_score[2] * MinMax / 2

    pos_inflow_rmse = pos_score[4] * MinMax / 2
    pos_inflow_mae = pos_score[5] * MinMax / 2

    pos_outflow_mape, pos_inflow_mape = test_mape((y_pos[:, :2, :, :] * minmax + minmax) / 2., (pos_pred * minmax + minmax) / 2.)

    neg_score = model.evaluate(x_neg, y_neg, batch_size=batch_size, verbose=0)
    neg_pred = model.predict(x_neg, batch_size=batch_size)

    neg_outflow_rmse = neg_score[1] * MinMax / 2
    neg_outflow_mae = neg_score[2] * MinMax / 2

    neg_inflow_rmse = neg_score[4] * MinMax / 2
    neg_inflow_mae = neg_score[5] * MinMax / 2

    neg_outflow_mape, neg_inflow_mape = test_mape((y_neg[:, :2, :, :] * minmax + minmax) / 2., (neg_pred * minmax + minmax) / 2.)

    print('-----{0} Test Score-----'.format(pos_name), end=' ')
    # print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f}'.format(
    #     pos_outflow_rmse, pos_outflow_mae, pos_inflow_rmse, pos_inflow_mae))
    print(
        'Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Pickup_MAPE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f} | Dropoff_MAPE: {:04f}'.format(
            pos_outflow_rmse, pos_outflow_mae, pos_outflow_mape, pos_inflow_rmse, pos_inflow_mae, pos_inflow_mape))

    print('-----{0} Test Score-----'.format(neg_name), end=' ')
    # print('Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f}'.format(
    #     neg_outflow_rmse, neg_outflow_mae, neg_inflow_rmse, neg_inflow_mae))
    print(
        'Pickup_RMSE: {:04f} | Pickup_MAE: {:04f} | Pickup_MAPE: {:04f} | Dropoff_RMSE: {:04f} | Dropoff_MAE: {:04f} | Dropoff_MAPE: {:04f}'.format(
            neg_outflow_rmse, neg_outflow_mae, neg_outflow_mape, neg_inflow_rmse, neg_inflow_mae, neg_inflow_mape))

    print('finish')


setproctitle.setproctitle('BJMobile DSTN+ResPlus @ ZiqianLin')  # from V1707

iter = 0
if args.dataset == 'nycbike' or args.dataset == 'BikeNYC':
    X_train, T_train, Y_train, X_test, T_test, Y_test, MinMax, Max, Min, len_total, len_test, number_of_skip_hours = \
        load_npy_data('Data/bike_flow.npy', len_test, len_closeness, len_period, len_trend, T_closeness,
                      T_period, T_trend)
    X_train = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=1)
    X_test = np.concatenate((X_test[0], X_test[1], X_test[2]), axis=1)
    T_train, T_test = get_Time('20160701', len_total, len_test, number_of_skip_hours)
    iter = 6
elif args.dataset == 'nyctaxi' or  args.dataset == 'TaxiNYC':
    X_train, T_train, Y_train, X_test, T_test, Y_test, MinMax, Max, Min, len_total, len_test, number_of_skip_hours = \
        load_npy_data('Data/taxi_flow.npy', len_test, len_closeness, len_period, len_trend, T_closeness,
                      T_period, T_trend)
    X_train = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=1)
    X_test = np.concatenate((X_test[0], X_test[1], X_test[2]), axis=1)
    T_train, T_test = get_Time('20150101', len_total, len_test, number_of_skip_hours)
    iter = 2
elif args.dataset == 'taxibj' or args.dataset == 'TaxiBJ':
    X_train, T_train, Y_train, X_test, T_test, Y_test, MinMax, Max, Min, len_total, len_test, number_of_skip_hours = \
        load_taxibj('Data/BJ13_M32x32_T30_InOut.h5', len_test, len_closeness, len_period, len_trend, T_closeness,
                    T_period, T_trend, T)
    T_train, T_test = get_Time('20130101', len_total, len_test, number_of_skip_hours)
    H, W, channel = 32, 32, 2
    X_train = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=1)
    X_test = np.concatenate((X_test[0], X_test[1], X_test[2]), axis=1)

Y_train = np.concatenate([Y_train, X_train], axis=1)
Y_test = np.concatenate([Y_test, X_test], axis=1)

# filename = './outputs/' + args.dataset + '.npz'
#
# if os.path.exists(filename):
#     data = np.load(filename)
#     preds = data['preds']
#     labels = Y_test
# else:
#     from Network.disentangle import Disentangle_Network
#     print("***** conv_model *****")
#     model = Disentangle_Network(batch_size=batch_size, H=H, W=W,
#                                 channel=channel, c=len_closeness, p=len_period, t=len_trend,
#                                 conv=64, mu_dim=128, R_N=2,
#                                 plus=8, rate=1, drop=0.1, lr=lr)
#
#     print('***** evaluate *****')
#     file_conv = '/root/JianyangQin/MUSE-Net/Exps/MUSE-Net_' + args.dataset + '_best.hdf5'
#     # file_conv = '/root/JianyangQin/MUSE-Net/Experiment/Model/MUSE-Net_eval_model_' + args.dataset + '.hdf5'
#     model.load_weights(file_conv)
#
#     Pickup_RMSE = np.zeros([1, 1])
#     Pickup_MAE = np.zeros([1, 1])
#     Pickup_MAPE = np.zeros([1, 1])
#     Dropoff_RMSE = np.zeros([1, 1])
#     Dropoff_MAE = np.zeros([1, 1])
#     Dropoff_MAPE = np.zeros([1, 1])
#
#     test_score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
#     print('Test  score:', end=' ')
#     np.set_printoptions(precision=6, suppress=True)
#     print(np.array(test_score))
#
#     Pickup_RMSE[0, 0] = test_score[1] * MinMax / 2
#     Pickup_MAE[0, 0] = test_score[2] * MinMax / 2
#     Pickup_MAPE[0, 0] = test_score[3] * 100
#     Dropoff_RMSE[0, 0] = test_score[4] * MinMax / 2
#     Dropoff_MAE[0, 0] = test_score[5] * MinMax / 2
#     Dropoff_MAPE[0, 0] = test_score[6] * 100
#
#     for_show = np.concatenate([Pickup_RMSE, Pickup_MAE, Pickup_MAPE, Dropoff_RMSE, Dropoff_MAE, Dropoff_MAPE], axis=1)
#     np.set_printoptions(precision=4, suppress=True)
#     print('RMSE  MAE  MAPE')
#     print(for_show)
#
#     preds = model.predict(X_test, batch_size=batch_size)
#     labels = Y_test
#     np.savez(filename, preds=preds, labels=labels)
#
#
# test_period(preds, labels, T_test, MinMax, get_WeekdayPeriod, pos_name='Weekday', neg_name='Weekend', viz=False)
#
# test_period(preds, labels, T_test, MinMax, get_PeekPeriod, pos_name='Peek', neg_name='Nonpeak', viz=False)


from Network.disentangle import Disentangle_Network
print("***** conv_model *****")
model = Disentangle_Network(batch_size=batch_size, H=H, W=W,
                            channel=channel, c=len_closeness, p=len_period, t=len_trend,
                            conv=64, mu_dim=128, R_N=2,
                            plus=8, rate=1, drop=0.1, lr=lr)

print('***** evaluate *****')
file_conv = '/root/JianyangQin/MUSE-Net/Exps/MUSE-Net_' + args.dataset + '_best.hdf5'
# file_conv = '/root/JianyangQin/MUSE-Net/Experiment/Model/MUSE-Net_eval_model_' + args.dataset + '.hdf5'
model.load_weights(file_conv)

evaluate_period(model, X_test, Y_test, T_test, MinMax, get_WeekdayPeriod, pos_name='Weekday', neg_name='Weekend')

evaluate_period(model, X_test, Y_test, T_test, MinMax, get_PeekPeriod, pos_name='Peek', neg_name='Nonpeak')


