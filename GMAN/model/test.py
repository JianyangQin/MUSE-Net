import torch
import time
import math
import numpy as np
from utils.utils_ import log_string, metric
from utils.utils_ import load_data

def test(device, args, log):
    (trainX, trainTE, trainY, valX, valTE, valY, testX, testTE,
     testY, SE, mean, std) = load_data(args)

    mean, std = np.asarray(mean), np.asarray(std)
    trainY, valY, testY = np.asarray(trainY), np.asarray(valY), np.asarray(testY)

    num_train, _, num_vertex, _ = trainX.shape
    num_val = valX.shape[0]
    num_test = testX.shape[0]
    train_num_batch = math.ceil(num_train / args.batch_size)
    val_num_batch = math.ceil(num_val / args.batch_size)
    test_num_batch = math.ceil(num_test / args.batch_size)

    model = torch.load(args.model_file)

    # test model
    log_string(log, '**** testing model ****')
    log_string(log, 'loading model from %s' % args.model_file)
    model = torch.load(args.model_file)
    log_string(log, 'model restored!')
    log_string(log, 'evaluating...')

    with torch.no_grad():

        trainPred = []
        for batch_idx in range(train_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_train, (batch_idx + 1) * args.batch_size)

            X = trainX[start_idx: end_idx]
            TE = trainTE[start_idx: end_idx]
            X, TE = X.to(device), TE.to(device)

            pred_batch = model(X, TE)
            trainPred.append(pred_batch.detach().cpu().numpy())
            del X, TE, pred_batch
        trainPred = np.concatenate(trainPred, axis=0)
        trainPred = trainPred * std + mean

        valPred = []
        for batch_idx in range(val_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_val, (batch_idx + 1) * args.batch_size)

            X = valX[start_idx: end_idx]
            TE = valTE[start_idx: end_idx]
            X, TE = X.to(device), TE.to(device)

            pred_batch = model(X, TE)
            valPred.append(pred_batch.detach().cpu().numpy())
            del X, TE, pred_batch
        valPred = np.concatenate(valPred, axis=0)
        valPred = valPred * std + mean

        testPred = []
        start_test = time.time()
        for batch_idx in range(test_num_batch):
            start_idx = batch_idx * args.batch_size
            end_idx = min(num_test, (batch_idx + 1) * args.batch_size)

            X = testX[start_idx: end_idx]
            TE = testTE[start_idx: end_idx]
            X, TE = X.to(device), TE.to(device)

            pred_batch = model(X, TE)
            testPred.append(pred_batch.detach().cpu().numpy())
            del X, TE, pred_batch
        testPred = np.concatenate(testPred, axis=0)
        testPred = testPred * std + mean
    end_test = time.time()

    log_string(log, 'testing time: %.1fs' % (end_test - start_test))

    # outflow
    train_mae_outflow, train_rmse_outflow, train_mape_outflow = metric(trainPred[:, 0, :, 0:1], trainY[:, 0, :, 0:1])
    val_mae_outflow, val_rmse_outflow, val_mape_outflow = metric(valPred[:, 0, :, 0:1], valY[:, 0, :, 0:1])
    test_mae_outflow, test_rmse_outflow, test_mape_outflow = metric(testPred[:, 0, :, 0:1], testY[:, 0, :, 0:1])
    log_string(log, 'Outflow         MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae_outflow, train_rmse_outflow, train_mape_outflow * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae_outflow, val_rmse_outflow, val_mape_outflow * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae_outflow, test_rmse_outflow, test_mape_outflow * 100))

    # inflow
    train_mae_inflow, train_rmse_inflow, train_mape_inflow = metric(trainPred[:, 0, :, 1:2], trainY[:, 0, :, 1:2])
    val_mae_inflow, val_rmse_inflow, val_mape_inflow = metric(valPred[:, 0, :, 1:2], valY[:, 0, :, 1:2])
    test_mae_inflow, test_rmse_inflow, test_mape_inflow = metric(testPred[:, 0, :, 1:2], testY[:, 0, :, 1:2])
    log_string(log, 'Inflow          MAE\t\tRMSE\t\tMAPE')
    log_string(log, 'train            %.2f\t\t%.2f\t\t%.2f%%' %
               (train_mae_inflow, train_rmse_inflow, train_mape_inflow * 100))
    log_string(log, 'val              %.2f\t\t%.2f\t\t%.2f%%' %
               (val_mae_inflow, val_rmse_inflow, val_mape_inflow * 100))
    log_string(log, 'test             %.2f\t\t%.2f\t\t%.2f%%' %
               (test_mae_inflow, test_rmse_inflow, test_mape_inflow * 100))

    test_score = [test_mae_outflow, test_rmse_outflow, test_mape_outflow, test_mae_inflow, test_rmse_inflow, test_mape_inflow]

    return trainPred, valPred, testPred, test_score
