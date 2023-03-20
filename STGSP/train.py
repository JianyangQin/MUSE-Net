import argparse
import os

import torch
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.optim as optim
from models.stgsp import STGSP
from models.metrics import masked_rmse_test, masked_mae_test, masked_mape_test, masked_mae
from data.dataset import load_data
from data.DataConfiguration import DataConfigurationBikeNYC, DataConfigurationTaxiNYC, DataConfigurationTaxiBJ
import numpy as np
from time import time
import shutil

seed = 666

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

def compute_val_loss(device, model, val_loader, criterion, missing_value, limit=None):
    '''
    for rnn, compute mean loss on validation set
    :param net: model
    :param val_loader: torch.utils.data.utils.DataLoader
    :param criterion: torch.nn.MSELoss
    :param sw: tensorboardX.SummaryWriter
    :param global_step: int, current global_step
    :param limit: int,
    :return: val_loss
    '''

    model.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            X, Y  = batch_data
            X = X.to(device)
            Y = Y.to(device)
            outputs = model(X)
            loss = criterion(outputs, Y)

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.5f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
    return validation_loss


def predict_and_save_results(device, model, data_loader, mmn, global_step, params_path, type):
    '''

    :param net: nn.Module
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param epoch: int
    :param _mean: (1, 1, 3, 1)
    :param _std: (1, 1, 3, 1)
    :param params_path: the path for saving the results
    :return:
    '''
    model.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():
        loader_length = len(data_loader)  # nb of batch

        trues = []
        preds = []  # 存储所有batch的output

        test_start = time()
        for batch_index, batch_data in enumerate(data_loader):
            X, Y = batch_data
            X = X.to(device)
            Y = Y.to(device)
            outputs = model(X)
            true = mmn.inverse_transform(Y.detach().cpu().numpy())
            pred = mmn.inverse_transform(outputs.detach().cpu().numpy())
            trues.append(true)
            preds.append(pred)

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))
        test_end = time()

        trues = np.concatenate(trues, 0)
        preds = np.concatenate(preds, 0)  # (batch, T', 1)

        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, prediction=preds, data_target_tensor=trues)

        # 计算误差
        excel_list = []
        mae_outflow = masked_mae_test(trues[:, 0, :, :].reshape(-1, 1), preds[:, 0, :, :].reshape(-1, 1))
        rmse_outflow = masked_rmse_test(trues[:, 0, :, :].reshape(-1, 1), preds[:, 0, :, :].reshape(-1, 1))
        mape_outflow = masked_mape_test(trues[:, 0, :, :].reshape(-1, 1), preds[:, 0, :, :].reshape(-1, 1))
        mae_inflow = masked_mae_test(trues[:, 1, :, :].reshape(-1, 1), preds[:, 1, :, :].reshape(-1, 1))
        rmse_inflow = masked_rmse_test(trues[:, 1, :, :].reshape(-1, 1), preds[:, 1, :, :].reshape(-1, 1))
        mape_inflow = masked_mape_test(trues[:, 1, :, :].reshape(-1, 1), preds[:, 1, :, :].reshape(-1, 1))

        print('Outflow RMSE: %.2f | MAE: %.2f | MAPE: %.2f' % (rmse_outflow, mae_outflow, mape_outflow))
        print('Inflow RMSE: %.2f | MAE: %.2f | MAPE: %.2f' % (rmse_inflow, mae_inflow, mape_inflow))

        excel_list.extend([rmse_outflow, mae_outflow, mape_outflow, rmse_inflow, mae_inflow, mape_inflow, test_start, test_end])
        print(excel_list)
    return excel_list

def train(args):
    # set gpu
    ctx = args.ctx
    os.environ['CUDA_VISIBLE_DEVICES'] = ctx
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA: ", use_cuda, device)

    # set args
    dataset = args.dataset
    start_epoch = args.start_epoch
    epochs = args.epochs
    learning_rate = args.learning_rate

    # set model save path
    params_path = os.path.join('exps', dataset)
    if (start_epoch == 0) and (not os.path.exists(params_path)):
        os.makedirs(params_path)
        print('create params directory %s' % (params_path))
    elif (start_epoch == 0) and (os.path.exists(params_path)):
        shutil.rmtree(params_path)
        os.makedirs(params_path)
        print('delete the old one and create params directory %s' % (params_path))
    elif (start_epoch > 0) and (os.path.exists(params_path)):
        print('train from params directory %s' % (params_path))
    else:
        raise SystemExit('Wrong type of model!')

    if args.dataset == 'BikeNYC':
        DataConfiguration = DataConfigurationBikeNYC
    elif args.dataset == 'TaxiNYC':
        DataConfiguration = DataConfigurationTaxiNYC
    else:
        DataConfiguration = DataConfigurationTaxiBJ
    print("l_c:", DataConfiguration.len_closeness, "l_p:", DataConfiguration.len_period, "l_t:",
          DataConfiguration.len_trend)
    set_seed(seed)

    # loda data
    dconf = DataConfiguration()
    train_loader, val_loader, test_loader, mmn = load_data(dconf)

    model = STGSP(dconf)
    # state_dict = torch.load("checkpoints/TaxiBJ/model_finetune.pth", map_location='cuda:0')
    # model.load_state_dict(state_dict)
    model = model.to(device)

    criterion = masked_mae
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)

    global_step = 0
    best_epoch = 0
    best_val_loss = np.inf
    missing_value = 0.0
    patience = 0

    if start_epoch > 0:
        params_filename = os.path.join(params_path, 'epoch_%s.params' % start_epoch)
        model.load_state_dict(torch.load(params_filename))
        print('start epoch:', start_epoch)
        print('load weight from: ', params_filename)

    # train model
    train_start = time()
    for epoch in range(start_epoch, epochs):
        params_filename = os.path.join(params_path, 'epoch_%s.params' % epoch)

        val_loss = compute_val_loss(device, model, val_loader, criterion, missing_value, epoch)
        if val_loss < best_val_loss:
            patience = 0
            best_val_loss = val_loss
            best_epoch = epoch
            torch.save(model.state_dict(), params_filename)
            print('save parameters to file: %s' % params_filename)
        else:
            patience += 1

        if patience > 30:
            break

        model.train()  # ensure dropout layers are in train mode

        for batch_index, batch_data in enumerate(train_loader):
            X, Y = batch_data
            X = X.to(device)
            Y = Y.to(device)
            optimizer.zero_grad()
            outputs = model(X)
            loss = criterion(outputs, Y)


            loss.backward()
            optimizer.step()
            training_loss = loss.item()

            global_step += 1

            if global_step % 1000 == 0:
                print('global step: %s, training loss: %.2f, time: %.2fs' % (global_step, training_loss, time() - train_start))

    train_end = time()
    print('best epoch:', best_epoch)

    # apply the best model on the test set
    predict(device, model, test_loader, mmn, best_epoch, params_path, 'test')


def predict(device, model, data_loader, mmn, global_step, params_path, type):
    '''

    :param global_step: int
    :param data_loader: torch.utils.data.utils.DataLoader
    :param data_target_tensor: tensor
    :param mean: (1, 1, 3, 1)
    :param std: (1, 1, 3, 1)
    :param type: string
    :return:
    '''

    params_filename = os.path.join(params_path, 'epoch_%s.params' % global_step)
    print('load weight from:', params_filename)

    model.load_state_dict(torch.load(params_filename))

    results = predict_and_save_results(device, model, data_loader, mmn, global_step, params_path, type)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", default='BikeNYC', type=str, help="BikeNYC, TaxiNYC, TaxiBJ")
    parser.add_argument("--ctx", default='0', type=str, help="gpu device")
    parser.add_argument("--start_epoch", default=0, type=int, help="resume")
    parser.add_argument("--learning_rate", default=0.001, type=float)
    parser.add_argument("--epochs", default=200, type=int)
    args = parser.parse_args()
    train(args)













