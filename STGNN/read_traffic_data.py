import numpy as np
import pandas as pd
import h5py

def seq2instance(data, P, Q):
    num_step, dims, nodes = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims, nodes))
    y = np.zeros(shape = (num_sample, Q, dims, nodes))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def time2instance(data, P, Q):
    num_step, dims = data.shape
    num_sample = num_step - P - Q + 1
    x = np.zeros(shape = (num_sample, P, dims))
    y = np.zeros(shape = (num_sample, Q, dims))
    for i in range(num_sample):
        x[i] = data[i : i + P]
        y[i] = data[i + P : i + P + Q]
    return x, y

def loadTrafficData(args):
    # Traffic
    if args.traffic_file.endswith('npy'):
        Traffic = np.load(args.traffic_file)
        Traffic = np.reshape(Traffic, (-1, Traffic.shape[1], Traffic.shape[2] * Traffic.shape[3]))
    else:
        df = h5py.File(args.traffic_file, 'r')
        Traffic = df['data'][:]
        Traffic = np.reshape(Traffic, (-1, Traffic.shape[1], Traffic.shape[2] * Traffic.shape[3]))

    # normalization
    traffic_max, traffic_min = np.max(Traffic), np.min(Traffic)
    Traffic = (2.0 * Traffic - (traffic_max + traffic_min)) / (traffic_max - traffic_min)

    T_trend, len_trend, len_test = args.T_trend, args.len_trend, args.len_test

    if len_trend > 0:
        number_of_skip_hours = T_trend * len_trend
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Traffic = Traffic[number_of_skip_hours - args.P - args.Q + 1:len(Traffic)]

    # batch_size, seq, flows, nodes
    allX, allY = seq2instance(Traffic, args.P, args.Q)
    num_step = allX.shape[0]
    train_steps = round(args.train_ratio * (num_step - len_test))
    test_steps = len_test
    val_steps = num_step - train_steps - test_steps
    trainX, trainY = allX[:train_steps], allY[:train_steps]
    valX, valY = allX[train_steps:train_steps + val_steps], allY[train_steps:train_steps + val_steps]
    testX, testY = allX[-test_steps:], allY[-test_steps:]

    # spatial embedding
    f = open(args.SE_file, mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = np.zeros(shape=(N, dims), dtype=np.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = temp[1:]

    return (trainX, trainY, valX, valY, testX, testY,
            SE, traffic_max, traffic_min)
