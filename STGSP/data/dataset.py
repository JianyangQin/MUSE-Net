import sys
sys.path.append("./")
import numpy as np
import h5py
import os
import math
import torch
import torch.utils.data as data
from data.minmax_normalization import MinMaxNormalization


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

def load_data(dconf):
    datapath = os.path.join(os.path.dirname(os.path.abspath(__file__)))
    filename = os.path.join(datapath, dconf.name, dconf.dataset)

    if dconf.dataset.endswith('npy'):
        all_data = np.load(filename)
    else:
        f = h5py.File(filename, 'r')
        all_data = f['data'].value
        timestamps = f['date'].value
        f.close()
        all_data, _ = remove_incomplete_days(all_data, timestamps, dconf.T)

    len_total, feature, map_height, map_width = all_data.shape
    print('all_data shape: ', all_data.shape)
    mmn = MinMaxNormalization()
    mmn.fit(all_data)
    print('max=', mmn.max, ' min=', mmn.min)

    all_data = mmn.transform(all_data)

    if dconf.len_trend > 0:
        number_of_skip_hours = dconf.T_trend * dconf.len_trend
    elif dconf.len_period > 0:
        number_of_skip_hours = dconf.T_period * dconf.len_period
    elif dconf.len_closeness > 0:
        number_of_skip_hours = dconf.T_closeness * dconf.len_closeness
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:len_total]
    len_train = round((len(Y) - dconf.len_test) * dconf.train_ratio)
    len_val = len(Y) - len_train - dconf.len_test

    if dconf.len_closeness > 0:
        X_closeness = all_data[number_of_skip_hours - dconf.T_closeness:len_total - dconf.T_closeness]
        for i in range(dconf.len_closeness - 1):
            X_closeness = np.concatenate(
                (X_closeness, all_data[number_of_skip_hours - dconf.T_closeness * (2 + i):len_total - dconf.T_closeness * (2 + i)]),
                axis=1)
    if dconf.len_period > 0:
        X_period = all_data[number_of_skip_hours - dconf.T_period:len_total - dconf.T_period]
        for i in range(dconf.len_period - 1):
            X_period = np.concatenate(
                (X_period, all_data[number_of_skip_hours - dconf.T_period * (2 + i):len_total - dconf.T_period * (2 + i)]), axis=1)
    if dconf.len_trend > 0:
        X_trend = all_data[number_of_skip_hours - dconf.T_trend:len_total - dconf.T_trend]
        for i in range(dconf.len_trend - 1):
            X_trend = np.concatenate(
                (X_trend, all_data[number_of_skip_hours - dconf.T_trend * (2 + i):len_total - dconf.T_trend * (2 + i)]), axis=1)


    X_closeness_train = X_closeness[:len_train]
    X_period_train = X_period[:len_train]
    X_trend_train = X_trend[:len_train]

    X_closeness_val = X_closeness[len_train:len_train + len_val]
    X_period_val = X_period[len_train:len_train + len_val]
    X_trend_val = X_trend[len_train:len_train + len_val]

    X_closeness_test = X_closeness[-dconf.len_test:]
    X_period_test = X_period[-dconf.len_test:]
    X_trend_test = X_trend[-dconf.len_test:]

    X_train = np.concatenate([X_closeness_train, X_period_train, X_trend_train], axis=1)
    X_val = np.concatenate([X_closeness_val, X_period_val, X_trend_val], axis=1)
    X_test = np.concatenate([X_closeness_test, X_period_test, X_trend_test], axis=1)

    Y_train = Y[:len_train]
    Y_val = Y[len_train:len_train + len_val]
    Y_test = Y[-dconf.len_test:]

    print('len_train=' + str(X_closeness_train.shape[0]))
    print('len_val=' + str(X_closeness_val.shape[0]))
    print('len_test =' + str(X_closeness_test.shape[0]))

    # ------- train_loader -------
    X_train = torch.from_numpy(X_train).type(torch.FloatTensor)
    Y_train = torch.from_numpy(Y_train).type(torch.FloatTensor)
    train_dataset = torch.utils.data.TensorDataset(X_train, Y_train)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=dconf.batch_size, shuffle=False)

    # ------- val_loader -------
    X_val = torch.from_numpy(X_val).type(torch.FloatTensor)
    Y_val = torch.from_numpy(Y_val).type(torch.FloatTensor)
    val_dataset = torch.utils.data.TensorDataset(X_val, Y_val)
    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=dconf.batch_size, shuffle=False)

    # ------- test_loader -------
    X_test = torch.from_numpy(X_test).type(torch.FloatTensor)
    Y_test = torch.from_numpy(Y_test).type(torch.FloatTensor)
    test_dataset = torch.utils.data.TensorDataset(X_test, Y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=dconf.batch_size, shuffle=False)

    return train_loader, val_loader, test_loader, mmn

if __name__ == '__main__':
    class DataConfigurationTaxiBJ():
        name = 'TaxiBJ'
        dataset = 'BJ13_M32x32_T30_InOut.h5'

        train_ratio = 0.9

        batch_size = 32

        len_closeness = 3
        len_period = 4
        len_trend = 4

        T = 24 * 2
        T_closeness = 1
        T_period = T
        T_trend = T * 7

        days_test = 20
        len_test = T * days_test

        dim_flow = 2
        dim_h = 32
        dim_w = 32

    train_loader, val_loader, test_loader, mmn = load_data(DataConfigurationTaxiBJ())

    print('train batch len = {}'.format(len(train_loader)))
    print('val batch len = {}'.format(len(val_loader)))
    print('test batch len = {}'.format(len(test_loader)))
    