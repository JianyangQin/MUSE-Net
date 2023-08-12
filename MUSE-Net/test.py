# from __future__ import print_function
from Network.disentangle import Disentangle_Network
from Data.read_nyc import load_npy_data
from Data.read_taxibj import load_taxibj
from keras.callbacks import ModelCheckpoint, EarlyStopping
from Network.metrics import metrics
import tensorflow as tf
import numpy as np
import argparse
import random
import os

parser = argparse.ArgumentParser(description='MUSE-Net')
parser.add_argument('--dataset', type=str, default='BikeNYC', help='BikeNYC, TaxiNYC or TaxiBJ')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--epoch', type=int, default=350)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--iterate', type=int, default=1)
parser.add_argument('--train_ratio', type=float, default=0.9)
parser.add_argument('--days_test', type=int, default=20)
parser.add_argument('--len_closeness', type=int, default=3)
parser.add_argument('--len_period', type=int, default=4)
parser.add_argument('--len_trend', type=int, default=4)
parser.add_argument('--feat_dim', type=int, default=64)
parser.add_argument('--mu_dim', type=int, default=128)

args = parser.parse_args()
print(args)

# for GPU in Lab
device = args.device
os.environ["CUDA_VISIBLE_DEVICES"] = device
import setproctitle

setproctitle.setproctitle('MUSE-Net Start! @ Jianyang Qin')

from keras import backend as K

K.set_image_data_format('channels_first')

# hyperparameters
epoch = args.epoch                  # number of epoch at training stage
batch_size = args.batch_size        # batch size
train_ratio = args.train_ratio      # train ratio
lr = 0.0002                         # learning rate

days_test = args.days_test
len_closeness = args.len_closeness  # length of closeness dependent sequence
len_period = args.len_period        # length of peroid dependent sequence
len_trend = args.len_trend          # length of trend dependent sequence

feat_dim = args.feat_dim            # feature dimension
mu_dim = args.mu_dim                # distribution dimension

class CustomStopper(EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)


if __name__ == "__main__":
    setproctitle.setproctitle('MUSE-Net @ Jianyang Qin')  # from V1707

    if args.dataset == 'BikeNYC':
        H, W, channel = 10, 20, 2   # grid size
        T = 24 * 2                  # number of time intervals in one day
        len_test = T * days_test
        T_closeness, T_period, T_trend = 1, T, T * 7

        X_train, Y_train, X_test, Y_test, MM, Max, Min = \
            load_npy_data('Data/bike_flow.npy', len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend)
    elif args.dataset == 'TaxiNYC':
        H, W, channel = 10, 20, 2  # grid size
        T = 24 * 2  # number of time intervals in one day
        len_test = T * days_test
        T_closeness, T_period, T_trend = 1, T, T * 7

        X_train, Y_train, X_test, Y_test, MM, Max, Min = \
            load_npy_data('Data/taxi_flow.npy', len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend)
    elif args.dataset == 'TaxiBJ':
        H, W, channel = 32, 32, 2  # grid size
        T = 24 * 2  # number of time intervals in one day
        len_test = T * days_test
        T_closeness, T_period, T_trend = 1, T, T * 7

        X_train, Y_train, X_test, Y_test, MM, Max, Min = \
            load_taxibj('Data/BJ13_M32x32_T30_InOut.h5', len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend, T)
    else:
        raise ValueError('dataset does not exist')

    X_train = np.concatenate((X_train[0], X_train[1], X_train[2]), axis=1)
    X_test = np.concatenate((X_test[0], X_test[1], X_test[2]), axis=1)

    Y_train = np.concatenate([Y_train, X_train], axis=1)
    Y_test = np.concatenate([Y_test, X_test], axis=1)

    import time

    Pickup_RMSE = np.zeros([1, 1])
    Pickup_MAE = np.zeros([1, 1])
    Pickup_MAPE = np.zeros([1, 1])
    Dropoff_RMSE = np.zeros([1, 1])
    Dropoff_MAE = np.zeros([1, 1])
    Dropoff_MAPE = np.zeros([1, 1])

    exps_path = 'Exps/'
    exps_files = os.listdir(exps_path)
    
    results, filenames = [], []
    
    model = Disentangle_Network(batch_size=batch_size, H=H, W=W,
                                        channel=channel, c=len_closeness, p=len_period, t=len_trend,
                                        feat_dim=feat_dim, conv=feat_dim, mu_dim=mu_dim,
                                        R_N=2, plus=8, rate=1, drop=0.1, lr=lr)

    for file in exps_files:
        if (args.dataset in file) and (file.endswith('hdf5')):
            file_conv = os.path.join(exps_path, file)

            # val
            print('=' * 10)
            print('***** evaluate *****')
            val_start = time.time()
            model.load_weights(file_conv)

            train_score = model.evaluate(X_train, Y_train, batch_size=batch_size, verbose=0)
            val_end = time.time()

            print('Outflow:    RMSE    MAE | Inflow:    RMSE    MAE')
            print('Train score:', end=' ')
            np.set_printoptions(precision=6, suppress=True)
            print(np.array(train_score))


            # test
            test_start = time.time()
            test_score = model.evaluate(X_test, Y_test, batch_size=batch_size, verbose=0)
            test_end = time.time()

            
            print('Test  score:', end=' ')
            np.set_printoptions(precision=6, suppress=True)
            print(np.array(test_score))

            Pickup_RMSE[0, 0] = test_score[1] * MM / 2
            Pickup_MAE[0, 0] = test_score[2] * MM / 2
            Dropoff_RMSE[0, 0] = test_score[4] * MM / 2
            Dropoff_MAE[0, 0] = test_score[5] * MM / 2

            for_show = np.concatenate([Pickup_RMSE, Pickup_MAE, Dropoff_RMSE, Dropoff_MAE], axis=1)
            np.set_printoptions(precision=4, suppress=True)
            print('Outflow:    RMSE    MAE | Inflow:    RMSE    MAE')
            print(for_show)
            
            results.append(for_show)
            filenames.append(file)

    len_results = len(results)
    len_files = len(filenames)
    
    if len_results == len_files:
        for i in range(len_files):
            res = results[i]
            filename = filenames[i]    
            print('{} Outflow RMSE: {:04f}, MAE: {:04f} | Inflow RMSE: {:04f}, MAE:{:04f}'.format(filename, res[0, 0], res[0, 1], res[0, 2], res[0, 3]))
