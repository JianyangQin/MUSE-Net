from __future__ import print_function
from load_data import load_npy_data, load_taxibj, generate_adjacency_matrix
import csv
import keras
from keras.layers import Input, Reshape, Conv3D, Dense, Flatten
from keras.models import Model
from keras.optimizers import Adam
from metrics import evaluate_performance
from keras.regularizers import l2
import numpy as np
from modifyGCNlayer import GraphConvolution1
from keras.callbacks import ModelCheckpoint, EarlyStopping
import time
import argparse
import os
import random
import tensorflow as tf
begintime = time.time()

parser = argparse.ArgumentParser(description='ConvGCN')
parser.add_argument('--dataset', type=str, default='TaxiBJ', help='BikeNYC, TaxiNYC or TaxiBJ')
parser.add_argument('--device', type=str, default='0')
parser.add_argument('--epoch', type=int, default=500)
parser.add_argument('--batch_size', type=int, default=64)
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

# hyperparameters
epoch = args.epoch                  # number of epoch at training stage
batch_size = args.batch_size        # batch size
train_ratio = args.train_ratio      # train ratio
lr = 0.0002                         # learning rate

days_test = args.days_test
len_closeness = args.len_closeness  # length of closeness dependent sequence
len_period = args.len_period        # length of peroid dependent sequence
len_trend = args.len_trend          # length of trend dependent sequence

iterate_num = 1

def seed_tensorflow(seed):
    os.environ['PYTHONHASHSEED'] = '0'
    random.seed(seed)
    np.random.seed(seed)
    tf.set_random_seed(seed)

class CustomStopper(EarlyStopping):
    # add argument for starting epoch
    def __init__(self, monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto', start_epoch=40):
        super().__init__(monitor=monitor, min_delta=min_delta, patience=patience, verbose=verbose, mode=mode)
        self.start_epoch = start_epoch

    def on_epoch_end(self, epoch, logs=None):
        if epoch > self.start_epoch:
            super().on_epoch_end(epoch, logs)

if __name__ == "__main__":
    seed_tensorflow(777)

    # ------------------------------ Data ------------------------------#
    if args.dataset == 'BikeNYC':
        T = 24 * 2
        T_closeness, T_period, T_trend = 1, T, T * 7
        len_test = days_test * T
        X_train, Y_train, X_val, Y_val, X_test, Y_test, MM, Max, Min = \
            load_npy_data('data/bike_flow.npy', train_ratio, len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend)
        H, W, channel = 10, 20, 2
        adjacency = generate_adjacency_matrix(H, W)
    elif args.dataset == 'TaxiNYC':
        T = 24 * 2
        T_closeness, T_period, T_trend = 1, T, T * 7
        len_test = days_test * T
        X_train, Y_train, X_val, Y_val, X_test, Y_test, MM, Max, Min = \
            load_npy_data('data/taxi_flow.npy', train_ratio, len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend)
        H, W, channel = 10, 20, 2
        adjacency = generate_adjacency_matrix(H, W)
    elif args.dataset == 'TaxiBJ':
        T = 24 * 2
        T_closeness, T_period, T_trend = 1, T, T * 7
        len_test = days_test * T
        X_train, Y_train, X_val, Y_val, X_test, Y_test, MM, Max, Min = \
            load_taxibj('data/BJ13_M32x32_T30_InOut.h5', train_ratio, len_test, len_closeness, len_period, len_trend, T_closeness,
                          T_period, T_trend, T)
        H, W, channel = 32, 32, 2
        adjacency = generate_adjacency_matrix(H, W)
    else:
        raise ValueError('dataset does not exist')

    X_train_outflow, X_train_inflow = X_train[:, :, :, 0], X_train[:, :, :, 0]
    X_val_outflow, X_val_inflow = X_val[:, :, :, 0], X_val[:, :, :, 0]
    X_test_outflow, X_test_inflow = X_test[:, :, :, 0], X_test[:, :, :, 0]
    num_of_vertices = H * W


    #------------------------------ Model ------------------------------#
    input1 = Input(shape=(X_train_outflow.shape[1], X_train_outflow.shape[2]))
    out1 = GraphConvolution1(15, adj=adjacency, activation='relu', kernel_regularizer=l2(5e-4))(input1)
    out1 = Reshape((num_of_vertices, 5, 3, 1), input_shape=(num_of_vertices, 15))(out1)

    # X_train_2,X_test_2 exit data
    input2 = Input(shape=(X_train_inflow.shape[1], X_train_inflow.shape[2]), name='input2')
    out2 = GraphConvolution1(15, adj=adjacency, activation='relu', kernel_regularizer=l2(5e-4))(input2)
    out2 = Reshape((num_of_vertices, 5, 3, 1), input_shape=(num_of_vertices, 15))(out2)

    out = keras.layers.concatenate([out1, out2], axis=4)

    out = Conv3D(16, kernel_size=3, padding='same', activation='relu')(out)

    out = Flatten()(out)
    out = Dense(num_of_vertices*2)(out)
    out = Reshape((num_of_vertices, 2), input_shape=(num_of_vertices*2,))(out)

    model = Model(inputs=[input1, input2], outputs=[out])

    model.compile(loss='mse', optimizer=Adam(lr=0.001))  # optimizer=RAdam(lr=0.001)
    print("finish compile")

    Pickup_RMSE = np.zeros([iterate_num, 1])
    Pickup_MAE = np.zeros([iterate_num, 1])
    Pickup_MAPE = np.zeros([iterate_num, 1])
    Dropoff_RMSE = np.zeros([iterate_num, 1])
    Dropoff_MAE = np.zeros([iterate_num, 1])
    Dropoff_MAPE = np.zeros([iterate_num, 1])

    for iterate_index in range(iterate_num):
        param_file = 'exps/Model_' + args.dataset + '_' + str(iterate_index) + '.h5'
        output_file = 'exps/Model_' + args.dataset + '_' + str(iterate_index) + '.npz'

        model.load_weights(param_file)
        Y_test_pred = model.predict([X_test_outflow, X_test_inflow], verbose=1)
        Y_test_pred = Y_test_pred.reshape(-1, num_of_vertices, 2)
        Y_test_pred = (Y_test_pred * MM + MM) / 2.

        Y_test_true = (Y_test * MM + MM) / 2.

        np.savez(output_file, preds = Y_test_pred, labels = Y_test_true)

        outflow_rmse, outflow_mae, outflow_mape, inflow_rmse, inflow_mae, inflow_mape = evaluate_performance(Y_test_true, Y_test_pred)  #


        Pickup_RMSE[iterate_index, 0] = outflow_rmse
        Pickup_MAE[iterate_index, 0] = outflow_mae
        Pickup_MAPE[iterate_index, 0] = outflow_mape
        Dropoff_RMSE[iterate_index, 0] = inflow_rmse
        Dropoff_MAE[iterate_index, 0] = inflow_mae
        Dropoff_MAPE[iterate_index, 0] = inflow_mape

        for_show = np.concatenate([Pickup_RMSE, Pickup_MAE, Pickup_MAPE, Dropoff_RMSE, Dropoff_MAE, Dropoff_MAPE], axis=1)
        np.set_printoptions(precision=4, suppress=True)
        print('RMSE     MAE     MAPE')
        print(for_show)
