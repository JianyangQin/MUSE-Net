import torch
import numpy as np
import argparse
import time
import util
from engine import trainer
import os
import json
import random

seed = 66

def set_seed(seed):
    torch.manual_seed(seed)  # cpu
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

def main(args):
    device = torch.device(args.device)
    config = json.load(open("data/config.json", "r"))[args.dataset]
    args.days = config["num_slots"]  # number of timeslots in a day which depends on the dataset
    args.num_nodes = config["num_nodes"]  # number of nodes
    args.normalization = config["normalization"]  # method of normalization which depends on the dataset
    args.data_dir = config["data_dir"]  # directory of data
    dataloader = util.load_dataset(args.data_dir, args.batch_size, args.batch_size, args.batch_size, days=args.days,
                                   sequence=args.seq_length, in_seq=args.in_len)
    scaler = dataloader['scaler']

    print(args)
    start_epoch = 1
    engine = trainer(scaler, args.in_dim, args.in_len, args.seq_length, args.num_nodes, args.nhid, args.dropout, args.normalization,
                     args.learning_rate, args.weight_decay, device, days=args.days, dims=args.dims, order=args.order)

    print("start training...", flush=True)
    his_loss = []
    val_time = []
    train_time = []
    count = 0

    for i in range(start_epoch, args.epochs + 1):
        # train
        train_loss = []
        train_mae_outflow, train_mae_inflow = [], []
        train_mape_outflow, train_mape_inflow = [], []
        train_rmse_outflow, train_rmse_inflow = [], []
        tt1 = time.time()
        dataloader['train_loader'].shuffle()
        for itera, (x, y, ind) in enumerate(dataloader['train_loader'].get_iterator()):
            trainx = torch.Tensor(x).to(device)
            trainx = trainx.transpose(1, 3)
            trainy = torch.Tensor(y).to(device)
            trainy = trainy.transpose(1, 3)
            metrics = engine.train(trainx, trainy, ind)
            train_loss.append((metrics[0] + metrics[3]) / 2.)
            train_mae_outflow.append(metrics[0])
            train_mape_outflow.append(metrics[1])
            train_rmse_outflow.append(metrics[2])
            train_mae_inflow.append(metrics[3])
            train_mape_inflow.append(metrics[4])
            train_rmse_inflow.append(metrics[5])

        tt2 = time.time()
        train_time.append(tt2 - tt1)
        # validate
        valid_loss = []
        valid_mae_outflow, valid_mae_inflow = [], []
        valid_mape_outflow, valid_mape_inflow = [], []
        valid_rmse_outflow, valid_rmse_inflow = [], []

        s1 = time.time()
        for itera, (x, y, ind) in enumerate(dataloader['val_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            testy = torch.Tensor(y).to(device)
            testy = testy.transpose(1, 3)
            metrics = engine.eval(testx, testy, ind)
            valid_loss.append((metrics[0] + metrics[3]) / 2.)
            valid_mae_outflow.append(metrics[0])
            valid_mape_outflow.append(metrics[1])
            valid_rmse_outflow.append(metrics[2])
            valid_mae_inflow.append(metrics[3])
            valid_mape_inflow.append(metrics[4])
            valid_rmse_inflow.append(metrics[5])
        s2 = time.time()
        log = 'Epoch: {:03d}, Inference Time: {:.4f} secs'
        print(log.format(i, (s2 - s1)))
        val_time.append(s2 - s1)
        mtrain_loss = np.mean(train_loss)
        mtrain_mae = np.mean(train_mae_outflow + train_mae_inflow)
        mtrain_mape = np.mean(train_mape_outflow + train_mape_inflow)
        mtrain_rmse = np.mean(train_rmse_outflow + train_rmse_inflow)

        mvalid_loss = np.mean(valid_loss)
        mvalid_mae = np.mean(valid_mae_outflow + valid_mae_inflow)
        mvalid_mape = np.mean(valid_mape_outflow + valid_mape_inflow)
        mvalid_rmse = np.mean(valid_rmse_outflow + valid_rmse_inflow)

        # early stopping
        if len(his_loss) > 0 and mvalid_loss < np.min(his_loss):
            count = 0
        else:
            count += 1
            print(f"no improve for {count} epochs")
        his_loss.append(mvalid_loss)

        log = 'Epoch: {:03d}, Train MAE: {:.4f}, Train MAPE: {:.4f}, Train RMSE: {:.4f},' \
              ' Valid MAE: {:.4f}, Valid MAPE: {:.4f}, Valid RMSE: {:.4f}, Training Time: {:.4f}/epoch'
        print(log.format(i, mtrain_mae, mtrain_mape, mtrain_rmse, mvalid_mae, mvalid_mape, mvalid_rmse, (tt2 - tt1)),
              flush=True)
        torch.save(engine.model.state_dict(),
                   os.path.join(args.save, "epoch_" + str(i) + "_" + str(round(float(mvalid_loss), 2)) + ".pth"))

        # test
        outputs = []
        realy = torch.Tensor(dataloader['y_test']).to(device)
        realy = realy.transpose(1, 3)
        for itera, (x, y, ind) in enumerate(dataloader['test_loader'].get_iterator()):
            testx = torch.Tensor(x).to(device)
            testx = testx.transpose(1, 3)
            with torch.no_grad():
                preds = engine.model(testx, ind)
                preds = preds.transpose(1, 3)
            outputs.append(preds)
        yhat = torch.cat(outputs, dim=0)
        yhat = yhat[:realy.size(0), ...]

        pred = scaler.inverse_transform(yhat)
        real = realy
        metrics = util.metric(pred, real)
        log = 'Evaluate best model on test data,' \
              ' Outflow MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f} | Inflow MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
        print(log.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))

        if count >= 30:
            break
    print("Average Training Time: {:.4f} secs/epoch".format(np.mean(train_time)))
    print("Average Inference Time: {:.4f} secs".format(np.mean(val_time)))
    print("Accumulate Training Time: {:.4f} secs".format(np.sum(train_time)))

    # final test
    bestid = np.argmin(his_loss)
    engine.model.load_state_dict(torch.load(
        os.path.join(args.save, "epoch_" + str(bestid + start_epoch)
                     + "_" + str(round(float(his_loss[int(bestid)]), 2)) + ".pth")))

    outputs = []
    realy = torch.Tensor(dataloader['y_test']).to(device)
    realy = realy.transpose(1, 3)

    for itera, (x, y, ind) in enumerate(dataloader['test_loader'].get_iterator()):
        testx = torch.Tensor(x).to(device)
        testx = testx.transpose(1, 3)
        with torch.no_grad():
            preds = engine.model(testx, ind)
            preds = preds.transpose(1, 3)
        outputs.append(preds)

    yhat = torch.cat(outputs, dim=0)
    yhat = yhat[:realy.size(0), ...]

    print("Training finished")
    print("The valid loss on best model is", str(round(float(his_loss[int(bestid)]), 4)))

    pred = scaler.inverse_transform(yhat[:, :, :, :])
    real = realy[:, :, :, :]
    metrics = util.metric(pred, real)
    log = 'Evaluate best model on test data,' \
          ' Outflow MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f} | Inflow MAE: {:.4f}, MAPE: {:.4f}, RMSE: {:.4f}'
    print(log.format(metrics[0], metrics[1], metrics[2], metrics[3], metrics[4], metrics[5]))


    torch.save(engine.model.state_dict(),
               os.path.join(args.save, "exp" + str(args.expid) +
                            "_best_" + str(round(float(his_loss[int(bestid)]), 2)) + ".pth"))

    np.savez(os.path.join(args.save, "output.npz"), preds=pred.detach().cpu().numpy(), labels=real.detach().cpu().numpy())

    return np.asarray(metrics)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--device', type=str, default='cuda:0', help='')
    parser.add_argument('--dataset', type=str, default='TaxiNYC', help='data path')
    parser.add_argument('--seq_length', type=int, default=1, help='output length')
    parser.add_argument('--in_len', type=int, default=11, help='input length')
    parser.add_argument('--nhid', type=int, default=64, help='')
    parser.add_argument('--in_dim', type=int, default=2, help='inputs dimension')
    parser.add_argument('--batch_size', type=int, default=64, help='batch size')
    parser.add_argument('--learning_rate', type=float, default=0.001, help='learning rate')
    parser.add_argument('--dropout', type=float, default=0.3, help='dropout rate')
    parser.add_argument('--weight_decay', type=float, default=0.0001, help='weight decay rate')
    parser.add_argument('--epochs', type=int, default=300, help='')
    parser.add_argument('--print_every', type=int, default=50, help='')
    parser.add_argument('--runs', type=int, default=1, help='number of experiments')
    parser.add_argument('--expid', type=int, default=1, help='experiment id')
    parser.add_argument('--iden', type=str, default='', help='identity')
    parser.add_argument('--dims', type=int, default=32, help='dimension of embeddings for dynamic graph')
    parser.add_argument('--order', type=int, default=2, help='order of graph convolution')

    args = parser.parse_args()
    args.save = os.path.join('exps/', os.path.basename(args.dataset) + args.iden)
    os.makedirs(args.save, exist_ok=True)
    t1 = time.time()
    metric = []

    for i in range(args.runs):
        args.expid = i
        set_seed(seed)
        metric.append(main(args))
        t2 = time.time()
        print("Total time spent: {:.4f}".format(t2 - t1))

