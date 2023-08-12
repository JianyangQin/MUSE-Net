import argparse
import os
from time import time

import torch
import numpy as np

from model.rnn import RNN
from model.seq2seq import Seq2Seq
from data.dataset import load_data
from data.data_configuration import DataConfigurationBikeNYC, DataConfigurationTaxiNYC, DataConfigurationTaxiBJ
from lib.metrics import masked_mae_test, masked_rmse_test, masked_mape_test


def predict_and_save_results(device, model, data_loader, mmn):

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

        # 计算误差
        excel_list = []
        mae_outflow = masked_mae_test(trues[:, :, :, 0].reshape(-1, 1), preds[:, :, :, 0].reshape(-1, 1))
        rmse_outflow = masked_rmse_test(trues[:, :, :, 0].reshape(-1, 1), preds[:, :, :, 0].reshape(-1, 1))
        mape_outflow = masked_mape_test(trues[:, :, :, 0].reshape(-1, 1), preds[:, :, :, 0].reshape(-1, 1))
        mae_inflow = masked_mae_test(trues[:, :, :, 1].reshape(-1, 1), preds[:, :, :, 1].reshape(-1, 1))
        rmse_inflow = masked_rmse_test(trues[:, :, :, 1].reshape(-1, 1), preds[:, :, :, 1].reshape(-1, 1))
        mape_inflow = masked_mape_test(trues[:, :, :, 1].reshape(-1, 1), preds[:, :, :, 1].reshape(-1, 1))

        print('Outflow RMSE: %.2f | MAE: %.2f | MAPE: %.2f' % (rmse_outflow, mae_outflow, mape_outflow))
        print('Inflow RMSE: %.2f | MAE: %.2f | MAPE: %.2f' % (rmse_inflow, mae_inflow, mape_inflow))

        excel_list.extend([rmse_outflow, mae_outflow, mape_outflow, rmse_inflow, mae_inflow, mape_inflow, test_start, test_end])
        print(excel_list)
    return excel_list

def test(args):
    # set gpu
    ctx = args.ctx
    os.environ['CUDA_VISIBLE_DEVICES'] = ctx
    use_cuda = torch.cuda.is_available()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    print("CUDA: ", use_cuda, device)

    # set args
    model_name = args.model_name

    if args.dataset == 'BikeNYC':
        DataConfiguration = DataConfigurationBikeNYC
    elif args.dataset == 'TaxiNYC':
        DataConfiguration = DataConfigurationTaxiNYC
    else:
        DataConfiguration = DataConfigurationTaxiBJ
    print("l_c:", DataConfiguration.len_closeness, "l_p:", DataConfiguration.len_period, "l_t:",
          DataConfiguration.len_trend)

    # loda data
    dconf = DataConfiguration()
    train_loader, val_loader, test_loader, mmn = load_data(dconf)

    # load model
    if model_name == 'RNN':
        model = RNN(device, args.input_seq, args.output_seq, args.input_dim, args.hidden_dim, args.num_of_layers)
    elif model_name == 'Seq2Seq':
        model = Seq2Seq(device, args.input_seq, args.output_seq, args.input_dim, args.hidden_dim, args.hidden_dim,
                        args.input_dim, args.num_of_layers)
    else:
        raise ValueError('model error!')
    model = model.to(device)

    params_filename = os.path.join(args.save_path, args.model_name, args.dataset, args.dataset + '_best_epoch.params')
    print('load weight from:', params_filename)

    model.load_state_dict(torch.load(params_filename))

    # predict
    results = predict_and_save_results(device, model, test_loader, mmn)

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_name", default='RNN', type=str, choices=["RNN", "Seq2Seq"],
                        help="select RNN or Seq2Seq model")
    parser.add_argument("--dataset", default='BikeNYC', type=str, choices=["BikeNYC", "TaxiNYC", "TaxiBJ"],
                        help="BikeNYC, TaxiNYC, TaxiBJ")
    parser.add_argument("--ctx", default='2', type=str, help="gpu device")
    parser.add_argument("--save_path", default='exps', type=str, help="saved model path")
    parser.add_argument("--input_seq", default=11, type=int)
    parser.add_argument("--output_seq", default=1, type=int)
    parser.add_argument("--input_dim", default=2, type=int)
    parser.add_argument("--hidden_dim", default=32, type=int)
    parser.add_argument("--num_of_layers", default=3, type=int)
    args = parser.parse_args()
    test(args)