import pandas as pd
import numpy as np
import torch
import h5py

def load_data(dataset, args):
    # Traffic
    if dataset.endswith('npy'):
        all_data = np.load(dataset)
        max, min = np.max(all_data), np.min(all_data)
    else:
        df = h5py.File(dataset, 'r')
        all_data = df['data'][:]
        max, min = np.max(all_data), np.min(all_data)

    len_total = len(all_data)
    T_closeness, T_period, T_trend = 1, args.T, args.T * 7

    if args.len_trend > 0:
        number_of_skip_hours = T_trend * args.len_trend
    elif args.len_period > 0:
        number_of_skip_hours = T_period * args.len_period
    elif args.len_closeness > 0:
        number_of_skip_hours = T_closeness * args.len_closeness
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:len_total]
    len_train = round((len(Y) - args.len_test) * args.train_ratio)
    len_val = len(Y) - len_train - args.len_test

    if args.len_closeness > 0:
        X_closeness = all_data[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(args.len_closeness - 1):
            X_closeness = np.concatenate(
                (X_closeness, all_data[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    if args.len_period > 0:
        X_period = all_data[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(args.len_period - 1):
            X_period = np.concatenate(
                (X_period, all_data[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]), axis=1)
    if args.len_trend > 0:
        X_trend = all_data[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(args.len_trend - 1):
            X_trend = np.concatenate(
                (X_trend, all_data[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]), axis=1)


    X_closeness_train = X_closeness[:len_train]
    X_period_train = X_period[:len_train]
    X_trend_train = X_trend[:len_train]

    X_closeness_val = X_closeness[len_train:len_train + len_val]
    X_period_val = X_period[len_train:len_train + len_val]
    X_trend_val = X_trend[len_train:len_train + len_val]

    X_closeness_test = X_closeness[-args.len_test:]
    X_period_test = X_period[-args.len_test:]
    X_trend_test = X_trend[-args.len_test:]

    trainX = np.concatenate([X_closeness_train, X_period_train, X_trend_train], axis=1)
    valX = np.concatenate([X_closeness_val, X_period_val, X_trend_val], axis=1)
    testX = np.concatenate([X_closeness_test, X_period_test, X_trend_test], axis=1)

    trainY = Y[:len_train]
    valY = Y[len_train:len_train + len_val]
    testY = Y[-args.len_test:]

    # batch_size, seq, H * W, 2
    trainX = trainX.reshape(trainX.shape[0], trainX.shape[1], -1)
    trainX = trainX.reshape(trainX.shape[0], -1, 2, trainX.shape[-1])
    trainX = np.transpose(trainX, (0, 1, 3, 2))
    trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], -1)
    trainY = np.expand_dims(np.transpose(trainY, (0, 2, 1)), 1)

    valX = valX.reshape(valX.shape[0], valX.shape[1], -1)
    valX = valX.reshape(valX.shape[0], -1, 2, valX.shape[-1])
    valX = np.transpose(valX, (0, 1, 3, 2))
    valY = valY.reshape(valY.shape[0], valY.shape[1], -1)
    valY = np.expand_dims(np.transpose(valY, (0, 2, 1)), 1)

    testX = testX.reshape(testX.shape[0], testX.shape[1], -1)
    testX = testX.reshape(testX.shape[0], -1, 2, testX.shape[-1])
    testX = np.transpose(testX, (0, 1, 3, 2))
    testY = testY.reshape(testY.shape[0], testY.shape[1], -1)
    testY = np.expand_dims(np.transpose(testY, (0, 2, 1)), 1)

    # batch_size, his_seq + pred_seq, H * W, 2
    trainX = np.concatenate([trainX, trainY], axis=1)
    valX = np.concatenate([valX, valY], axis=1)
    testX = np.concatenate([testX, testY], axis=1)

    # normalize x
    trainX = (2.0 * trainX - (max + min)) / (max - min)
    valX = (2.0 * valX - (max + min)) / (max - min)
    testX = (2.0 * testX - (max + min)) / (max - min)

    return (trainX, valX, testX, max, min)
