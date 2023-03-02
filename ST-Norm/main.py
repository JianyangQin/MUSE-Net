# -*- coding: utf-8 -*-
from __future__ import print_function
import os
import time

import numpy as np
import math
from models.Wavenet import Wavenet
from utils.data_utils import *
from utils.math_utils import *
from utils.tester import model_inference
import torchvision
import torch
import torch.optim as optim
import torch.nn as nn
from torch.autograd import Variable
import argparse

seed = 777

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

np.random.seed(seed)  # for reproducibility
# torch.backends.cudnn.benchmark = True

set_seed(seed)


batch_size = 8  # batch size
test_batch_size = 48

lr = 0.0001  # learning rate


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type=str, default='BikeNYC', help='BikeNYC, TaxiNYC or TaxiBJ')
parser.add_argument('--version', type=int, default=0)
parser.add_argument('--model', type=str, default='wavenet')
parser.add_argument('--mode', type=str, default='train')
parser.add_argument('--snorm', type=int, default=1)
parser.add_argument('--tnorm', type=int, default=1)
parser.add_argument('--n_his', type=int, default=11)
parser.add_argument('--n_pred', type=int, default=1)
parser.add_argument('--n_layers', type=int, default=4)
parser.add_argument('--hidden_channels', type=int, default=16)
parser.add_argument('--T', type=int, default=48)
parser.add_argument('--train_ratio', type=float, default=0.9)
parser.add_argument('--len_test', type=int, default=960)
parser.add_argument('--len_closeness', type=int, default=3)
parser.add_argument('--len_period', type=int, default=4)
parser.add_argument('--len_trend', type=int, default=4)
parser.add_argument('--epochs', type=int, default=500)
args = parser.parse_args()

dataset_name = args.dataset
snorm_bool = bool(args.snorm)
tnorm_bool = bool(args.tnorm)
n_his = args.n_his
n_pred = args.n_pred
n_layers = args.n_layers
hidden_channels = args.hidden_channels
version = args.version


def train(model, dataset, n):
    target_n = "s{}_t{}_hc{}_l{}_his{}_pred{}_v{}".format(args.snorm, args.tnorm, args.hidden_channels, n_layers, n_his, n_pred, args.version)
    target_fname = '{}_{}_{}'.format(args.model, args.dataset, target_n)
    target_model_path = os.path.join('exps', '{}.h5'.format(target_fname))
    print('=' * 10)
    print("training model...")

    print("releasing gpu memory....")
    model.train()
    criterion = nn.MSELoss()
    optimizer = optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)
    torch.cuda.empty_cache()

    min_rmse = 1000
    # min_val = min_va_val = np.array([4e1, 1e5, 1e5] * n_pred)
    stop = 0
    nb_epoch = args.epochs

    train_start = time.time()
    for epoch in range(nb_epoch):  # loop over the dataset multiple times
        model.train()
        epoch_start = time.time()
        for j, x_batch in enumerate(gen_batch(dataset.get_data('train'), batch_size, dynamic_batch=True, shuffle=True)):
            xh = x_batch[:, 0: n_his]
            y = x_batch[:, n_his:n_his + n_pred]
            xh = torch.tensor(xh, dtype=torch.float32).cuda()
            y = torch.tensor(y, dtype=torch.float32).cuda()
            model.zero_grad()
            pred = model(xh)
            loss = criterion(pred, y)
            loss.backward()
            nn.utils.clip_grad_norm_(model.parameters(), 10)
            optimizer.step()
        epoch_end = time.time()

        model.eval()
        val_outflow, val_inflow, test_outflow, test_inflow = model_inference(model, dataset, test_batch_size, n_his, n_pred, n)
        print(f'Epoch {epoch} {epoch_end - epoch_start}:')
        vo, vi, to, ti = val_outflow, val_inflow, test_outflow, test_inflow
        print(f'MAPE outflow:{vo[0]:7.3%}, inflow:{vi[0]:7.3%};  '
              f'MAE  outflow:{vo[1]:4.3f}, inflow:{vi[1]:4.3f};  '
              f'RMSE outflow:{vo[2]:6.3f}, inflow:{vi[2]:6.3f}.')

        total_rmse = (np.sum([vo[i*3+2] for i in range(n_pred)]) + np.sum([vi[i*3+2] for i in range(n_pred)])) / 2.0
        if total_rmse < min_rmse:
            torch.save(model.state_dict(), target_model_path)
            min_rmse = total_rmse
            stop = 0
        else:
            stop += 1
        if stop == 20:
            break
    train_end = time.time()
    model.load_my_state_dict(torch.load(target_model_path))

    test_start = time.time()
    val_outflow, val_inflow, test_outflow, test_inflow = model_inference(model, dataset, test_batch_size, n_his, n_pred, n)
    test_end = time.time()

    vo, vi, to, ti = val_outflow, val_inflow, test_outflow, test_inflow
    print('Best Results:')
    print(f'MAPE outflow:{to[0]:7.3%}, inflow:{ti[0]:7.3%};  '
        f'MAE  outflow:{to[1]:4.3f}, inflow:{ti[1]:4.3f};  '
        f'RMSE outflow:{to[2]:6.3f}, inflow:{ti[2]:6.3f}.')

    print(f'\n train time cost: {train_end - train_start}')
    print(f'\n test time cost: {test_end - test_start}')


def eval(model, dataset, n, versions):
    print('=' * 10)
    print("evaluating model...")
    vos, vis = [], []
    tos, tis = [], []
    for _v in versions:
        target_n = "s{}_t{}_hc{}_l{}_his{}_pred{}_v{}".format(args.snorm, args.tnorm, args.hidden_channels, n_layers, n_his, n_pred, _v)
        target_fname = '{}_{}_{}'.format(args.model, args.dataset, target_n)
        target_model_path = os.path.join('exps', '{}.h5'.format(target_fname))
        if os.path.isfile(target_model_path):
            model.load_my_state_dict(torch.load(target_model_path))
        else:
            print("file not exist")
            break
        val_outflow, val_inflow, test_outflow, test_inflow = model_inference(model, dataset, test_batch_size, n_his, n_pred, n)
        print(f'Version:{_v}')
        vo, vi, to, ti = val_outflow, val_inflow, test_outflow, test_inflow
        print(f'MAPE {to[0]:7.3%}, {ti[0]:7.3%};  '
              f'MAE  {to[1]:4.3f}, {ti[1]:4.3f};  '
              f'RMSE {to[2]:6.3f}, {ti[2]:6.3f}.')
        vos.append(vo)
        vis.append(vi)
        tos.append(to)
        tis.append(ti)
    vo, vi = np.array(vos).mean(axis=0), np.array(vis).mean(axis=0)
    to, ti = np.array(tos).mean(axis=0), np.array(tis).mean(axis=0)
    print(f'Overall:')
    print(f'MAPE {to[0]:7.3%}, {ti[0]:7.3%};  '
          f'MAE  {to[1]:4.3f}, {ti[1]:4.3f};  '
          f'RMSE {to[2]:6.3f}, {ti[2]:6.3f}.')


def main():
    # load data
    print("loading data...")
    if dataset_name == 'BikeNYC':
        n = 200
        dataset = traffic_data_gen('data/bike_flow.npy', args)
        in_dim = 2
    elif dataset_name == 'TaxiNYC':
        n = 200
        dataset = traffic_data_gen('data/taxi_flow.npy', args)
        in_dim = 2
    elif dataset_name == 'TaxiBJ':
        n = 1024
        dataset = traffic_data_gen('data/BJ13_M32x32_T30_InOut.h5', args)
        in_dim = 2
    else:
        raise ValueError('dataset does not exist')

    print('=' * 10)
    print("compiling model...")
    model = Wavenet(n, tnorm_bool=tnorm_bool, snorm_bool=snorm_bool, in_dim=in_dim,out_dim=n_pred, channels=hidden_channels, kernel_size=2, blocks=1, layers=n_layers).cuda()


    print('=' * 10)
    print("init model...")
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
        else:
            nn.init.uniform_(p)

    if args.mode == 'train':
        train(model, dataset, n)
    if args.mode == 'eval':
        eval(model, dataset, n, [0])

if __name__ == '__main__':
    main()
