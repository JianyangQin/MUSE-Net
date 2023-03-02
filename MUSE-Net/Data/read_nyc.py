import numpy as np
import h5py

class MM:
    def __init__(self, MM_max, MM_min):
        self.max = MM_max
        self.min = MM_min


def load_npy_data(dataset, train_ratio, len_test, len_closeness, len_period, len_trend, T_closeness=1, T_period=24, T_trend=24 * 7):
    all_data = np.load(dataset)
    len_total, feature, map_height, map_width = all_data.shape
    print('all_data shape: ', all_data.shape)
    mm = MM(np.max(all_data), np.min(all_data))
    print('max=', mm.max, ' min=', mm.min)

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
    len_train = round((len(Y) - len_test) * train_ratio)
    len_val = len(Y) - len_train - len_test

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

    X_closeness_train = X_closeness[:len_train]
    X_period_train = X_period[:len_train]
    X_trend_train = X_trend[:len_train]

    X_closeness_val = X_closeness[len_train:len_train+len_val]
    X_period_val = X_period[len_train:len_train+len_val]
    X_trend_val = X_trend[len_train:len_train+len_val]


    X_closeness_test = X_closeness[-len_test:]
    X_period_test = X_period[-len_test:]
    X_trend_test = X_trend[-len_test:]

    X_train = [X_closeness_train, X_period_train, X_trend_train]
    X_val = [X_closeness_val, X_period_val, X_trend_val]
    X_test = [X_closeness_test, X_period_test, X_trend_test]

    Y_train = Y[:len_train]
    Y_val = Y[len_train:len_train+len_val]
    Y_test = Y[-len_test:]


    print('len_train=' + str(len_train))
    print('len_val=' + str(len_val))
    print('len_test =' + str(len_test))

    return X_train, Y_train, X_val, Y_val, X_test, Y_test, mm.max - mm.min, mm.max, mm.min