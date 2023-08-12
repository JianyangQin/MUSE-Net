import os
import numpy as np
import h5py
import torch
import torch.utils.data
from sklearn.metrics import mean_absolute_error
from sklearn.metrics import mean_squared_error
from scipy.sparse.linalg import eigs
from .metrics import masked_mape_test,  masked_mae, masked_mse, masked_rmse, masked_mae_test, masked_rmse_test


def re_normalization(x, mean, std):
    x = x * std + mean
    return x


def max_min_normalization(x, _max, _min):
    x = 1. * (x - _min)/(_max - _min)
    x = x * 2. - 1.
    return x


def re_max_min_normalization(x, _max, _min):
    x = (x + 1.) / 2.
    x = 1. * x * (_max - _min) + _min
    return x


def get_adjacency_matrix(distance_df_filename, num_of_vertices, id_filename=None):
    '''
    Parameters
    ----------
    distance_df_filename: str, path of the csv file contains edges information

    num_of_vertices: int, the number of vertices

    Returns
    ----------
    A: np.ndarray, adjacency matrix

    '''
    if 'npy' in distance_df_filename:

        adj_mx = np.load(distance_df_filename)

        return adj_mx, None

    else:

        import csv

        A = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                     dtype=np.float32)

        distaneA = np.zeros((int(num_of_vertices), int(num_of_vertices)),
                            dtype=np.float32)

        if id_filename:

            with open(id_filename, 'r') as f:
                id_dict = {int(i): idx for idx, i in enumerate(f.read().strip().split('\n'))}  # 把节点id（idx）映射成从0开始的索引

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[id_dict[i], id_dict[j]] = 1
                    distaneA[id_dict[i], id_dict[j]] = distance
            return A, distaneA

        else:

            with open(distance_df_filename, 'r') as f:
                f.readline()
                reader = csv.reader(f)
                for row in reader:
                    if len(row) != 3:
                        continue
                    i, j, distance = int(row[0]), int(row[1]), float(row[2])
                    A[i, j] = 1
                    distaneA[i, j] = distance
            return A, distaneA


def scaled_Laplacian(W):
    '''
    compute \tilde{L}

    Parameters
    ----------
    W: np.ndarray, shape is (N, N), N is the num of vertices

    Returns
    ----------
    scaled_Laplacian: np.ndarray, shape (N, N)

    '''

    assert W.shape[0] == W.shape[1]

    D = np.diag(np.sum(W, axis=1))

    L = D - W

    lambda_max = eigs(L, k=1, which='LR')[0].real

    return (2 * L) / lambda_max - np.identity(W.shape[0])


def cheb_polynomial(L_tilde, K):
    '''
    compute a list of chebyshev polynomials from T_0 to T_{K-1}

    Parameters
    ----------
    L_tilde: scaled Laplacian, np.ndarray, shape (N, N)

    K: the maximum order of chebyshev polynomials

    Returns
    ----------
    cheb_polynomials: list(np.ndarray), length: K, from T_0 to T_{K-1}

    '''

    N = L_tilde.shape[0]

    cheb_polynomials = [np.identity(N), L_tilde.copy()]

    for i in range(2, K):
        cheb_polynomials.append(2 * L_tilde * cheb_polynomials[i - 1] - cheb_polynomials[i - 2])

    return cheb_polynomials


def load_data(DEVICE, traffic_file, batch_size, len_closeness, len_period, len_trend, T, days_test, train_ratio):
    # traffic flow
    if traffic_file.endswith('npy'):
        all_data = np.load(traffic_file)
    else:
        f = h5py.File(traffic_file, 'r')
        all_data = f['data'].value

    mean, std = np.mean(all_data), np.std(all_data)
    max, min = np.max(all_data), np.min(all_data)
    len_total = len(all_data)

    len_closeness, len_period, len_trend = len_closeness, len_period, len_trend
    T_closeness, T_period, T_trend = 1, T, T * 7
    len_test = days_test * T

    if len_trend > 0:
        number_of_skip_hours = T_trend * len_trend
    else:
        print("wrong")
    print('number_of_skip_hours:', number_of_skip_hours)

    Y = all_data[number_of_skip_hours:len_total]
    len_train = round((len(Y) - len_test) * train_ratio)
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

    train_x = np.concatenate([X_closeness_train, X_period_train, X_trend_train], axis=1)
    val_x = np.concatenate([X_closeness_val, X_period_val, X_trend_val], axis=1)
    test_x = np.concatenate([X_closeness_test, X_period_test, X_trend_test], axis=1)

    train_target = Y[:len_train]
    val_target = Y[len_train:len_train + len_val]
    test_target = Y[-len_test:]

    # normalize x
    train_x = (2.0 * train_x - (max + min)) / (max - min)
    val_x = (2.0 * val_x - (max + min)) / (max - min)
    test_x = (2.0 * test_x - (max + min)) / (max - min)

    # x: (batch_size, H * W, 2, seq), y: (batch_size, H * W, 2)
    train_x = train_x.reshape(train_x.shape[0], train_x.shape[1], -1)
    train_x = train_x.reshape(train_x.shape[0], -1, 2, train_x.shape[-1])
    train_x = np.transpose(train_x, (0, 3, 2, 1))
    train_target = train_target.reshape(train_target.shape[0], train_target.shape[1], -1)
    train_target = np.expand_dims(np.transpose(train_target, (0, 2, 1)), -1)

    val_x = val_x.reshape(val_x.shape[0], val_x.shape[1], -1)
    val_x = val_x.reshape(val_x.shape[0], -1, 2, val_x.shape[-1])
    val_x = np.transpose(val_x, (0, 3, 2, 1))
    val_target = val_target.reshape(val_target.shape[0], val_target.shape[1], -1)
    val_target = np.expand_dims(np.transpose(val_target, (0, 2, 1)), -1)

    test_x = test_x.reshape(test_x.shape[0], test_x.shape[1], -1)
    test_x = test_x.reshape(test_x.shape[0], -1, 2, test_x.shape[-1])
    test_x = np.transpose(test_x, (0, 3, 2, 1))
    test_target = test_target.reshape(test_target.shape[0], test_target.shape[1], -1)
    test_target = np.expand_dims(np.transpose(test_target, (0, 2, 1)), -1)

    # ------- train_loader -------
    train_x_tensor = torch.from_numpy(train_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    train_target_tensor = torch.from_numpy(train_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    train_dataset = torch.utils.data.TensorDataset(train_x_tensor, train_target_tensor)

    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=False)

    # ------- val_loader -------
    val_x_tensor = torch.from_numpy(val_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    val_target_tensor = torch.from_numpy(val_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    val_dataset = torch.utils.data.TensorDataset(val_x_tensor, val_target_tensor)

    val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=batch_size, shuffle=False)

    # ------- test_loader -------
    test_x_tensor = torch.from_numpy(test_x).type(torch.FloatTensor).to(DEVICE)  # (B, N, F, T)
    test_target_tensor = torch.from_numpy(test_target).type(torch.FloatTensor).to(DEVICE)  # (B, N, T)

    test_dataset = torch.utils.data.TensorDataset(test_x_tensor, test_target_tensor)

    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

    # print
    print('train:', train_x_tensor.size(), train_target_tensor.size())
    print('val:', val_x_tensor.size(), val_target_tensor.size())
    print('test:', test_x_tensor.size(), test_target_tensor.size())

    return train_loader, train_target_tensor, val_loader, val_target_tensor, test_loader, test_target_tensor, mean, std, max, min


def compute_val_loss_mstgcn(net, val_loader, criterion, masked_flag, missing_value, sw, epoch, limit=None):
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

    net.train(False)  # ensure dropout layers are in evaluation mode

    with torch.no_grad():

        val_loader_length = len(val_loader)  # nb of batch

        tmp = []  # 记录了所有batch的loss

        for batch_index, batch_data in enumerate(val_loader):
            encoder_inputs, labels = batch_data
            outputs = net(encoder_inputs)
            if masked_flag:
                loss = criterion(outputs, labels, missing_value)
            else:
                loss = criterion(outputs, labels)

            tmp.append(loss.item())
            if batch_index % 100 == 0:
                print('validation batch %s / %s, loss: %.2f' % (batch_index + 1, val_loader_length, loss.item()))
            if (limit is not None) and batch_index >= limit:
                break

        validation_loss = sum(tmp) / len(tmp)
        sw.add_scalar('validation_loss', validation_loss, epoch)
    return validation_loss



def predict_and_save_results_mstgcn(net, data_loader, data_target_tensor, global_step, metric_method,_mean, _std, params_path, type):
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
    net.train(False)  # ensure dropout layers are in test mode

    with torch.no_grad():

        data_target_tensor = data_target_tensor.cpu().numpy()

        loader_length = len(data_loader)  # nb of batch

        prediction = []  # 存储所有batch的output

        input = []  # 存储所有batch的input

        for batch_index, batch_data in enumerate(data_loader):

            encoder_inputs, labels = batch_data

            input.append(encoder_inputs[:, :, 0:1].cpu().numpy())  # (batch, T', 1)

            outputs = net(encoder_inputs)

            prediction.append(outputs.detach().cpu().numpy())

            if batch_index % 100 == 0:
                print('predicting data set batch %s / %s' % (batch_index + 1, loader_length))

        input = np.concatenate(input, 0)

        input = re_normalization(input, _mean, _std)

        prediction = np.concatenate(prediction, 0)  # (batch, T', 1)

        print('input:', input.shape)
        print('prediction:', prediction.shape)
        print('data_target_tensor:', data_target_tensor.shape)
        output_filename = os.path.join(params_path, 'output_epoch_%s_%s' % (global_step, type))
        np.savez(output_filename, input=input, prediction=prediction, data_target_tensor=data_target_tensor)

        # 计算误差
        excel_list = []
        prediction_length = prediction.shape[3]

        for i in range(prediction_length):
            assert data_target_tensor.shape[0] == prediction.shape[0]
            print('\ncurrent epoch: %s, predict %s points' % (global_step, i))

            inflow_mae = masked_mae_test(data_target_tensor[:, :, 1, i], prediction[:, :, 1, i])
            inflow_rmse = masked_rmse_test(data_target_tensor[:, :, 1, i], prediction[:, :, 1, i])
            inflow_mape = masked_mape_test(data_target_tensor[:, :, 1, i], prediction[:, :, 1, i], 10.) * 100
            outflow_mae = masked_mae_test(data_target_tensor[:, :, 0, i], prediction[:, :, 0, i])
            outflow_rmse = masked_rmse_test(data_target_tensor[:, :, 0, i], prediction[:, :, 0, i])
            outflow_mape = masked_mape_test(data_target_tensor[:, :, 0, i], prediction[:, :, 0, i], 10.) * 100

            print('MAE: outflow-%.2f inflow-%.2f' % (outflow_mae, inflow_mae))
            print('RMSE: outflow-%.2f inflow-%.2f' % (outflow_rmse, inflow_rmse))
            print('MAPE: outflow-%.2f inflow-%.2f' % (outflow_mape, inflow_mape))

        # print overall results
        print('\ncurrent epoch: %s, predict overall' % (global_step))

        inflow_mae = masked_mae_test(data_target_tensor[:, :, 1, :].reshape(-1, 1), prediction[:, :, 1, :].reshape(-1, 1))
        inflow_rmse = masked_rmse_test(data_target_tensor[:, :, 1, :].reshape(-1, 1), prediction[:, :, 1, :].reshape(-1, 1))
        inflow_mape = masked_mape_test(data_target_tensor[:, :, 1, :].reshape(-1, 1), prediction[:, :, 1, :].reshape(-1, 1), 10.) * 100
        outflow_mae = masked_mae_test(data_target_tensor[:, :, 0, :].reshape(-1, 1), prediction[:, :, 0, :].reshape(-1, 1))
        outflow_rmse = masked_rmse_test(data_target_tensor[:, :, 0, :].reshape(-1, 1), prediction[:, :, 0, :].reshape(-1, 1))
        outflow_mape = masked_mape_test(data_target_tensor[:, :, 0, :].reshape(-1, 1), prediction[:, :, 0, :].reshape(-1, 1), 10.) * 100

        print('MAE: outflow-%.2f inflow-%.2f' % (outflow_mae, inflow_mae))
        print('RMSE: outflow-%.2f inflow-%.2f' % (outflow_rmse, inflow_rmse))
        print('MAPE: outflow-%.2f inflow-%.2f' % (outflow_mape, inflow_mape))
        excel_list.extend([outflow_mae, outflow_rmse, outflow_mape, inflow_mae, inflow_rmse, inflow_mape])
        print(excel_list)

def generate_adjacency_matrix(col, row):
    adj_mx = np.zeros((col*row, col*row))
    for i in range(col * row):
        for j in range(col * row):
            if j == i:
                adj_mx[i][j] = 1.0
            elif (j == i + 1) and (j // col == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i - 1) and (j // col == i // col):
                adj_mx[i][j] = 1.0
            elif j == i - col:
                adj_mx[i][j] = 1.0
            elif j == i + col:
                adj_mx[i][j] = 1.0
            elif (j == i - col - 1) and ((j // col + 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i - col + 1) and ((j // col + 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i + col - 1) and ((j // col - 1) == i // col):
                adj_mx[i][j] = 1.0
            elif (j == i + col + 1) and ((j // col - 1) == i // col):
                adj_mx[i][j] = 1.0
            else:
                adj_mx[i][j] = 0.0
    return adj_mx
