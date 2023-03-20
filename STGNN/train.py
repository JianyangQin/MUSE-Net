from tqdm import tqdm
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import time
import argparse
import math

from utils import log_string
from read_traffic_data import loadTrafficData
from model import STGNN

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', type = str, default='BikeNYC',
                    help='BikeNYC, TaxiNYC or TaxiBJ')
parser.add_argument('--device', type=int, default = 0,
                    help = 'gpu device')
parser.add_argument('--max_epoch', type = int, default = 50,
                    help = 'epoch to run')
parser.add_argument('--batch_size', type = int, default = 16,
                    help = 'batch size')
parser.add_argument('--learning_rate', type=float, default = 0.001,
                    help = 'initial learning rate')
parser.add_argument('--train_ratio', type = float, default = 0.9,
                    help = 'training set [default : 0.7]')
parser.add_argument('--T_trend', type=int, default=336,
                    help='num of time intervals in one day * num of day intervals in trend = 48 * 7')
parser.add_argument('--len_trend', type=int, default=4,
                    help='len of trend sequence = 4')
parser.add_argument('--len_test', type=int, default=960,
                    help='num of test samples = 960')
parser.add_argument('--P', type = int, default = 12,
                    help = 'history steps')
parser.add_argument('--Q', type = int, default = 12,
                    help = 'prediction steps')
parser.add_argument('--L', type = int, default = 1,
                    help = 'number of STAtt Blocks')
parser.add_argument('--K', type = int, default = 4,
                    help = 'number of attention heads')
parser.add_argument('--d', type = int, default = 16,
                    help = 'dims of each head attention outputs')


args = parser.parse_args()

seed = 777

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

if args.dataset == 'BikeNYC':
    args.traffic_file = 'data/bike_flow.npy'
    args.SE_file = 'data/SE_BikeNYC.txt'
    args.model_file = 'BikeNYC'
    args.log_file ='exps/BikeNYC_log'
elif args.dataset == 'TaxiNYC':
    args.traffic_file = 'data/taxi_flow.npy'
    args.SE_file = 'data/SE_TaxiNYC.txt'
    args.model_file = 'TaxiNYC'
    args.log_file = 'exps/TaxiNYC_log'
elif args.dataset == 'TaxiBJ':
    args.traffic_file = 'data/BJ13_M32x32_T30_InOut.h5'
    args.SE_file = 'data/SE_TaxiBJ.txt'
    args.model_file = 'TaxiBJ'
    args.log_file = 'exps/TaxiBJ_log'
else:
    raise ValueError('dataset does not exist')

log = open(args.log_file, 'w')

# device = torch.device('cuda', args.device)
device = torch.device("cuda:"+str(args.device) if torch.cuda.is_available() else "cpu")

log_string(log, "loading data....")

set_seed(seed)
np.random.seed(seed)

trainX, trainY, valX, valY, testX, testY, _, mmax, mmin = loadTrafficData(args)

log_string(log, "loading end....")

def res(save_path, model, valX, valY, mmax, mmin):
    model.eval() # 评估模式, 这会关闭dropout
    # it = test_iter.get_iterator()
    num_val = valX.shape[0]
    pred = []
    label = []
    num_batch = math.ceil(num_val / args.batch_size)
    with torch.no_grad():
        for batch_idx in range(num_batch):
            if isinstance(model, torch.nn.Module):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(valX[start_idx : end_idx]).float().to(device)
                y = valY[start_idx : end_idx]

                y_hat = model(X)

                pred.append(y_hat.cpu().numpy())
                label.append(y)
    
    pred = np.concatenate(pred, axis = 0)
    label = np.concatenate(label, axis = 0)

    if save_path is not None:
        preds = (pred * (mmax - mmin) + (mmax - mmin)) / 2
        labels = (label * (mmax - mmin) + (mmax - mmin)) / 2
        np.savez(save_path, preds = preds, labels = labels)

    # print(pred.shape, label.shape)
    in_maes = []
    in_rmses = []
    in_mapes = []

    out_maes = []
    out_rmses = []
    out_mapes = []

    log_string(log,'\ninflow prediction:')

    in_pred, out_pred = pred[:, :, 0, :], pred[:, :, 1, :]
    in_label, out_label = label[:, :, 0, :], label[:, :, 1, :]
    for i in range(args.Q):
        in_mae, in_rmse, in_mape = metric(in_pred[:,i,:], in_label[:,i,:], mmax - mmin)
        in_maes.append(in_mae)
        in_rmses.append(in_rmse)
        in_mapes.append(in_mape)
        if i == 0:
            log_string(log, 'in step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i + 1, in_mae, in_rmse, in_mape))

    in_mae, in_rmse, in_mape = metric(in_pred, in_label, mmax - mmin)
    in_maes.append(in_mae)
    in_rmses.append(in_rmse)
    in_mapes.append(in_mape)

    log_string(log, '\noutflow prediction:')
    for i in range(args.Q):
        out_mae, out_rmse, out_mape = metric(out_pred[:,i,:], out_label[:,i,:], mmax - mmin)
        out_maes.append(out_mae)
        out_rmses.append(out_rmse)
        out_mapes.append(out_mape)
        if i == 0:
            log_string(log,'out step %d, mae: %.4f, rmse: %.4f, mape: %.4f' % (i+1, out_mae, out_rmse, out_mape))

    out_mae, out_rmse, out_mape = metric(out_pred, out_label, mmax - mmin)
    out_maes.append(out_mae)
    out_rmses.append(out_rmse)
    out_mapes.append(out_mape)

    return np.stack(in_maes, 0), np.stack(in_rmses, 0), np.stack(in_mapes, 0), np.stack(out_maes, 0), np.stack(out_rmses, 0), np.stack(out_mapes, 0)

def train(model, trainX, trainY, valX, valY, mmax, mmin):
    num_train = trainX.shape[0]
    min_loss = 10000000.0
    model.train()
    optimizer = torch.optim.Adam(model.parameters(),
                                     lr=args.learning_rate)
    # lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, milestones=[10, 15],
    #                                                         gamma=0.2)
    lr_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5,    
                                    verbose=False, threshold=0.001, threshold_mode='rel', cooldown=0, min_lr=2e-6, eps=1e-08)
    
    for epoch in tqdm(range(1,args.max_epoch+1)):
        model.train()
        train_l_sum, train_acc_sum, n, batch_count, start = 0.0, 0.0, 0, 0, time.time()
        permutation = np.random.permutation(num_train)
        trainX = trainX[permutation]
        trainY = trainY[permutation]
        num_batch = math.ceil(num_train / args.batch_size)
        with tqdm(total=num_batch) as pbar:
            for batch_idx in range(num_batch):
                start_idx = batch_idx * args.batch_size
                end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

                X = torch.from_numpy(trainX[start_idx : end_idx]).float().to(device)
                y = torch.from_numpy(trainY[start_idx : end_idx]).float().to(device)

                optimizer.zero_grad()

                y_hat = model(X)

                y_d = y
                y_hat_d = y_hat


                loss = _compute_loss(y, y_hat)

                loss.backward()
                nn.utils.clip_grad_norm_(model.parameters(), 5)
                optimizer.step()
                
                train_l_sum += loss.cpu().item()
                # print(f"\nbatch loss: {l.cpu().item()}")
                n += y.shape[0]
                batch_count += 1
                pbar.update(1)
        # lr = lr_scheduler.get_lr()
        log_string(log, 'epoch %d, lr %.6f, loss %.4f, time %.1f sec'
              % (epoch, optimizer.param_groups[0]['lr'], train_l_sum / batch_count, time.time() - start))

        in_mae, in_rmse, in_mape, out_mae, out_rmse, out_mape = res(None, model, valX, valY, mmax, mmin)
        # lr_scheduler.step()
        lr_scheduler.step(in_mae[-1])
        if in_mae[-1] < min_loss:
            min_loss = in_mae[-1]
            torch.save(model, 'exps/' + args.model_file)

def test(save_path, model, valX, valY, mmax, mmin):
    model = torch.load('exps/' + args.model_file)
    in_mae, in_rmse, in_mape, out_mae, out_rmse, out_mape = res(save_path, model, valX, valY, mmax, mmin)
    return in_mae, in_rmse, in_mape, out_mae, out_rmse, out_mape

def _compute_loss(y_true, y_predicted):
        return masked_mae(y_predicted, y_true, 0.0)

def masked_mae(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels!=null_val)
    mask = mask.float()
    mask /=  torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = torch.abs(preds-labels)
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)

def metric(pred, label, mm, threshold=10.0):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        pred = (pred * mm + mm) / 2.
        label = (label * mm + mm) / 2.

        mae = np.abs(np.subtract(pred, label)).astype(np.float32)

        rmse = np.square(mae)

        mae = np.mean(mae)
        rmse = np.sqrt(np.mean(rmse))

        mask = label > threshold
        if np.sum(mask) != 0:
            mape = np.mean(np.abs(label[mask] - pred[mask]).astype('float64') / label[mask])
    return mae, rmse, mape * 100

if __name__ == '__main__':
    in_maes, in_rmses, in_mapes, out_maes, out_rmses, out_mapes = [], [], [], [], [], []
    for i in range(1):
        log_string(log, "model constructed begin....")
        model = STGNN(device, 2, args.K*args.d, args.L, args.d).to(device)
        log_string(log, "model constructed end....")
        log_string(log, "train begin....")
        train_start = time.time()
        train(model, trainX, trainY, testX, testY, mmax, mmin)
        train_end = time.time()
        log_string(log, "train end....")
        test_start = time.time()
        in_mae, in_rmse, in_mape, out_mae, out_rmse, out_mape = test('./exps/' + args.dataset + '.npz', model, testX, testY, mmax, mmin)
        test_end = time.time()
        in_maes.append(in_mae)
        in_rmses.append(in_rmse)
        in_mapes.append(in_mape)
        out_maes.append(out_mae)
        out_rmses.append(out_rmse)
        out_mapes.append(out_mape)
    log_string(log, "\n\nresults:")
    in_maes = np.stack(in_maes, 1)
    in_rmses = np.stack(in_rmses, 1)
    in_mapes = np.stack(in_mapes, 1)
    out_maes = np.stack(out_maes, 1)
    out_rmses = np.stack(out_rmses, 1)
    out_mapes = np.stack(out_mapes, 1)

    log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f' % (0, in_maes[0].mean(), in_rmses[0].mean(), in_mapes[0].mean()))

    log_string(log, 'step %d, mae %.4f, rmse %.4f, mape %.4f' % (0, out_maes[0].mean(), out_rmses[0].mean(), out_mapes[0].mean()))


    file_txt = open('exps/STGNN_' + args.model_file + '.txt', 'w')

    np.set_printoptions(precision=4, suppress=True)
    print('MAE  RMSE  MAPE', file=file_txt)

    print('step %d, mae %.4f, rmse %.4f, mape %.4f' % (0, in_maes[0].mean(), in_rmses[0].mean(), in_mapes[0].mean()), file=file_txt)

    print('step %d, mae %.4f, rmse %.4f, mape %.4f' % (0, out_maes[0].mean(), out_rmses[0].mean(), out_mapes[0].mean()), file=file_txt)

    print('\n train_start: {0}, train_end: {1}, train_cost: {2}, test_start: {3}, test_end: {4}, test_cost: {5}'.format(train_start, train_end, train_end - train_start, test_start, test_end, test_end - test_start), file=file_txt)

    file_txt.close()