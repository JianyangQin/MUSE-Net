import numpy as np
# import pandas as pd
import h5py
import argparse
import os

def generate_adj(col, row):
    adj_mx = np.zeros((col * row, col * row))
    for i in range(col * row):
        for j in range(col * row):
            if j == i:
                adj_mx[i][j] = 0.0
            elif (j == i + 1) and (j // col == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i - 1) and (j // col == i // col):
                adj_mx[i][j] = 1.0
            elif j == i - col:
                adj_mx[i][j] = 1.0
            elif j == i + col:
                adj_mx[i][j] = 1.0
            elif (j == i - col - 1) and ((j // col + 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i - col + 1) and ((j // col + 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i + col - 1) and ((j // col - 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i + col + 1) and ((j // col - 1) == i // col):
                adj_mx[i][j] = 1.0
            else:
                adj_mx[i][j] = 0.0
    return adj_mx

def loadTrafficData(args):
    # Traffic
    if args.traffic_file.endswith('npy'):
        all_data = np.load(args.traffic_file)
        max, min = np.max(all_data), np.min(all_data)
    else:
        df = h5py.File(args.traffic_file, 'r')
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
    trainX = trainX.reshape(trainX.shape[0], 2, -1, trainX.shape[-1])
    trainX = np.transpose(trainX, (0, 2, 3, 1))
    trainY = trainY.reshape(trainY.shape[0], trainY.shape[1], -1)
    trainY = np.expand_dims(trainY, 2)
    trainY = np.transpose(trainY, (0, 2, 3, 1))

    valX = valX.reshape(valX.shape[0], valX.shape[1], -1)
    valX = valX.reshape(valX.shape[0], 2, -1, valX.shape[-1])
    valX = np.transpose(valX, (0, 2, 3, 1))
    valY = valY.reshape(valY.shape[0], valY.shape[1], -1)
    valY = np.expand_dims(valY, 2)
    valY = np.transpose(valY, (0, 2, 3, 1))

    testX = testX.reshape(testX.shape[0], testX.shape[1], -1)
    testX = testX.reshape(testX.shape[0], 2, -1, testX.shape[-1])
    testX = np.transpose(testX, (0, 2, 3, 1))
    testY = testY.reshape(testY.shape[0], testY.shape[1], -1)
    testY = np.expand_dims(testY, 2)
    testY = np.transpose(testY, (0, 2, 3, 1))

    return trainX, trainY, valX, valY, testX, testY

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--dataset', default='BikeNYC', help='traffic file load path')
    parser.add_argument('--flow', default='inflow', choices=['inflow', 'outflow'], help='traffic file load path')
    parser.add_argument('--train_ratio', type=float, default=0.9, help='training set [default : 0.9]')
    parser.add_argument('--T', type=int, default=48, help='num of time intervals in one day = 48')
    parser.add_argument('--T_trend', type=int, default=336, help='num of time intervals in one day * num of day intervals in trend = 48 * 7')
    parser.add_argument('--len_closeness', type=int, default=3, help='len of closeness sequence = 3')
    parser.add_argument('--len_period', type=int, default=4, help='len of period sequence = 4')
    parser.add_argument('--len_trend', type=int, default=4, help='len of trend sequence = 4')
    parser.add_argument('--len_test', type=int, default=960, help='num of test samples = 960')
    parser.add_argument('--P', type=int, default=11, help='history steps')
    parser.add_argument('--Q', type=int, default=1, help='prediction steps')
    args = parser.parse_args()
    if args.dataset == 'BikeNYC':
        args.traffic_file = 'data/BikeNYC/bike_flow.npy'
        num_of_col, num_of_row = 10, 20
        args.save_path = 'data/BikeNYC'
    elif args.dataset == 'TaxiNYC':
        args.traffic_file = 'data/TaxiNYC/taxi_flow.npy'
        num_of_col, num_of_row = 10, 20
        args.save_path = 'data/TaxiNYC'
    elif args.dataset == 'TaxiBJ':
        args.traffic_file = 'data/TaxiBJ/BJ13_M32x32_T30_InOut.h5'
        num_of_col, num_of_row = 32, 32
        args.save_path = 'data/TaxiBJ'
    else:
        raise ValueError('dataset does not exist')

    trainX, trainY, valX, valY, testX, testY = loadTrafficData(args)
    adj_mx = generate_adj(num_of_col, num_of_row)

    np.savez(os.path.join(args.save_path, 'train.npz'), x=trainX, y=trainY)
    np.savez(os.path.join(args.save_path, 'val.npz'), x=valX, y=valY)
    np.savez(os.path.join(args.save_path, 'test.npz'), x=testX, y=testY)
    np.savez(os.path.join(args.save_path, 'adj_mx.npz'), adj_mx=adj_mx)
