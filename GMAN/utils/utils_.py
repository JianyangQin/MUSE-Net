import pandas as pd
import torch
import matplotlib.pyplot as plt
from torch.utils.data import Dataset
import numpy as np
import h5py


# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)


def metric(pred, label, threshold=10.0):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)

        rmse = np.square(mae)

        mae = np.mean(mae)
        rmse = np.sqrt(np.mean(rmse))

        mask = label > threshold
        if np.sum(mask) != 0:
            mape = np.mean(np.abs(label[mask] - pred[mask]).astype('float64') / label[mask])
    return mae, rmse, mape


def load_data(args):
    # traffic flow
    if args.traffic_file.endswith('npy'):
        all_data = np.load(args.traffic_file)
    else:
        f = h5py.File(args.traffic_file, 'r')
        all_data = f['data'].value

    mean, std = np.mean(all_data), np.std(all_data)
    len_total = len(all_data)

    len_closeness, len_period, len_trend = args.len_closeness, args.len_period, args.len_trend
    T_closeness, T_period, T_trend = 1, args.T, args.T * 7
    len_test = args.days_test * args.T

    if len_trend > 0:
        number_of_skip_hours = T_trend * len_trend
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:len_total]
    len_train = round((len(Y) - len_test) * args.train_ratio)
    len_val = len(Y) - len_train - len_test

    if len_closeness > 0:
        X_closeness = all_data[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(len_closeness - 1):
            X_closeness = np.concatenate(
                (X_closeness,
                 all_data[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    if len_period > 0:
        X_period = all_data[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(len_period - 1):
            X_period = np.concatenate(
                (X_period, all_data[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]),
                axis=1)
    if len_trend > 0:
        X_trend = all_data[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(len_trend - 1):
            X_trend = np.concatenate(
                (X_trend, all_data[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]), axis=1)

    X_closeness_train = X_closeness[:len_train]
    X_period_train = X_period[:len_train]
    X_trend_train = X_trend[:len_train]

    X_closeness_val = X_closeness[len_train:len_train + len_val]
    X_period_val = X_period[len_train:len_train + len_val]
    X_trend_val = X_trend[len_train:len_train + len_val]

    X_closeness_test = X_closeness[-len_test:]
    X_period_test = X_period[-len_test:]
    X_trend_test = X_trend[-len_test:]

    trainX = np.concatenate([X_closeness_train, X_period_train, X_trend_train], axis=1)
    valX = np.concatenate([X_closeness_val, X_period_val, X_trend_val], axis=1)
    testX = np.concatenate([X_closeness_test, X_period_test, X_trend_test], axis=1)

    trainY = Y[:len_train]
    valY = Y[len_train:len_train + len_val]
    testY = Y[-len_test:]

    # Reshape

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

    # normalize x
    trainX = (trainX - mean) / std
    valX = (valX - mean) / std
    testX = (testX - mean) / std


    # temporal embedding
    freq = '30min'
    freq_delta_second = 60 * 30
    t = pd.date_range(args.start_time, periods=len_total, freq=freq)
    df = pd.DataFrame({"date": t})
    df['date'] = pd.to_datetime(df['date'])
    df.set_index("date", inplace=True)
    Time = df.index
    dayofweek = np.expand_dims(np.reshape(Time.weekday, newshape=(-1, 1)), 1)
    timeofday = (Time.hour * 3600 + Time.minute * 60 + Time.second) // freq_delta_second
    timeofday = np.expand_dims(np.reshape(timeofday, newshape=(-1, 1)), 1)
    Time = np.concatenate((dayofweek, timeofday), axis=-1)

    # train/val/test
    Y_Time = Time[number_of_skip_hours:len_total]

    if len_closeness > 0:
        TE_closeness = Time[number_of_skip_hours - T_closeness:len_total - T_closeness]
        for i in range(len_closeness - 1):
            TE_closeness = np.concatenate(
                (
                TE_closeness, Time[number_of_skip_hours - T_closeness * (2 + i):len_total - T_closeness * (2 + i)]),
                axis=1)
    if len_period > 0:
        TE_period = Time[number_of_skip_hours - T_period:len_total - T_period]
        for i in range(len_period - 1):
            TE_period = np.concatenate(
                (TE_period, Time[number_of_skip_hours - T_period * (2 + i):len_total - T_period * (2 + i)]), axis=1)
    if len_trend > 0:
        TE_trend = Time[number_of_skip_hours - T_trend:len_total - T_trend]
        for i in range(len_trend - 1):
            TE_trend = np.concatenate(
                (TE_trend, Time[number_of_skip_hours - T_trend * (2 + i):len_total - T_trend * (2 + i)]), axis=1)

    TE_closeness_train = TE_closeness[:len_train]
    TE_period_train = TE_period[:len_train]
    TE_trend_train = TE_trend[:len_train]

    TE_closeness_val = TE_closeness[len_train:len_train + len_val]
    TE_period_val = TE_period[len_train:len_train + len_val]
    TE_trend_val = TE_trend[len_train:len_train + len_val]

    TE_closeness_test = TE_closeness[-len_test:]
    TE_period_test = TE_trend[-len_test:]
    TE_trend_test = TE_trend[-len_test:]

    trainXT = np.concatenate([TE_closeness_train, TE_period_train, TE_trend_train], axis=1)
    valXT = np.concatenate([TE_closeness_val, TE_period_val, TE_trend_val], axis=1)
    testXT = np.concatenate([TE_closeness_test, TE_period_test, TE_trend_test], axis=1)

    trainYT = Y_Time[:len_train]
    valYT = Y_Time[len_train:len_train + len_val]
    testYT = Y_Time[-len_test:]

    trainTE = np.concatenate([trainXT, trainYT], axis=1)
    valTE = np.concatenate([valXT, valYT], axis=1)
    testTE = np.concatenate([testXT, testYT], axis=1)

    # spatial embedding
    f = open(args.SE_file, mode='r')
    lines = f.readlines()
    temp = lines[0].split(' ')
    N, dims = int(temp[0]), int(temp[1])
    SE = torch.zeros((N, dims), dtype=torch.float32)
    for line in lines[1:]:
        temp = line.split(' ')
        index = int(temp[0])
        SE[index] = torch.tensor([float(ch) for ch in temp[1:]])

    trainX, trainTE, trainY = torch.from_numpy(trainX).type(torch.FloatTensor), torch.from_numpy(trainTE).type(torch.FloatTensor), torch.from_numpy(trainY).type(torch.FloatTensor)
    valX, valTE, valY = torch.from_numpy(valX).type(torch.FloatTensor), torch.from_numpy(valTE).type(torch.FloatTensor), torch.from_numpy(valY).type(torch.FloatTensor)
    testX, testTE, testY = torch.from_numpy(testX).type(torch.FloatTensor), torch.from_numpy(testTE).type(torch.FloatTensor), torch.from_numpy(testY).type(torch.FloatTensor)
    mean, std = torch.from_numpy(np.asarray(mean)).type(torch.FloatTensor), torch.from_numpy(np.asarray(std)).type(torch.FloatTensor)


    print('train shape=' + str(trainX.shape))
    print('val shape=' + str(valX.shape))
    print('test shape=' + str(testX.shape))


    return (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE, testY,
            SE, mean, std)

    # dataset creation
class dataset(Dataset):
    def __init__(self, data_x, data_y):
        self.data_x = data_x
        self.data_y = data_y
        self.len = data_x.shape[0]

    def __getitem__(self, index):
        return self.data_x[index], self.data_y[index]

    def __len__(self):
        return self.len


# statistic model parameters
def count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# The following function can be replaced by 'loss = torch.nn.L1Loss()  loss_out = loss(pred, target)
def mae_loss(pred, label):
    mask = torch.ne(label, 0)
    mask = mask.type(torch.float32)
    mask /= torch.mean(mask)
    mask = torch.where(torch.isnan(mask), torch.tensor(0.0), mask)
    loss = torch.abs(torch.sub(pred, label))
    loss *= mask
    loss = torch.where(torch.isnan(loss), torch.tensor(0.0), loss)
    loss = torch.mean(loss)
    return loss


# plot train_val_loss
def plot_train_val_loss(train_total_loss, val_total_loss, file_path):
    plt.figure(figsize=(10, 5))
    plt.plot(range(1, len(train_total_loss) + 1), train_total_loss, c='b', marker='s', label='Train')
    plt.plot(range(1, len(val_total_loss) + 1), val_total_loss, c='r', marker='o', label='Validation')
    plt.legend(loc='best')
    plt.title('Train loss vs Validation loss')
    plt.savefig(file_path)


# plot test results
def save_test_result(trainPred, trainY, valPred, valY, testPred, testY):
    with open('./figure/test_results.txt', 'w+') as f:
        for l in (trainPred, trainY, valPred, valY, testPred, testY):
            f.write(list(l))
