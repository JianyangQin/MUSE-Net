import pickle
import numpy as np
import os
import torch


class DataLoader(object):
    def __init__(self, xs, ys, batch_size, begin=0, days=288, pad_with_last_sample=True):
        """
        :param xs:
        :param ys:
        :param batch_size:
        :param pad_with_last_sample: pad with the last sample to make number of samples divisible to batch_size.
        """
        self.batch_size = batch_size
        self.current_ind = 0
        if pad_with_last_sample:
            num_padding = (batch_size - (len(xs) % batch_size)) % batch_size
            x_padding = np.repeat(xs[-1:], num_padding, axis=0)
            y_padding = np.repeat(ys[-1:], num_padding, axis=0)
            xs = np.concatenate([xs, x_padding], axis=0)
            ys = np.concatenate([ys, y_padding], axis=0)
        self.size = len(xs)
        self.num_batch = int(self.size // self.batch_size)
        self.xs = xs
        self.ys = ys
        self.ind = np.arange(begin, begin + self.size)
        self.days = days

    def shuffle(self):
        permutation = np.random.permutation(self.size)
        xs, ys = self.xs[permutation], self.ys[permutation]
        self.ind = self.ind[permutation]
        self.xs = xs
        self.ys = ys

    def get_iterator(self):
        self.current_ind = 0

        def _wrapper():
            while self.current_ind < self.num_batch:
                start_ind = self.batch_size * self.current_ind
                end_ind = min(self.size, self.batch_size * (self.current_ind + 1))
                x_i = self.xs[start_ind: end_ind, ...]
                y_i = self.ys[start_ind: end_ind, ...]
                i_i = self.ind[start_ind: end_ind, ...] % self.days
                yield (x_i, y_i, i_i)
                self.current_ind += 1

        return _wrapper()


class StandardScaler():
    """
    Standard the input
    """

    def __init__(self, max, min, mean, std):
        self.max = max
        self.min = min
        self.mean = mean
        self.std = std

    def transform(self, data):
        return (2.0*data - (self.max + self.min)) / (self.max - self.min)

    def inverse_transform(self, data):
        return (data * (self.max - self.min) + (self.max + self.min)) / 2.0

def load_pickle(pickle_file):
    try:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f)
    except UnicodeDecodeError as e:
        with open(pickle_file, 'rb') as f:
            pickle_data = pickle.load(f, encoding='latin1')
    except Exception as e:
        print('Unable to load data ', pickle_file, ':', e)
        raise
    return pickle_data


def load_dataset(dataset_dir, batch_size, valid_batch_size=None, test_batch_size=None, days=288, sequence=12,
                 in_seq=12):
    data = {}
    for category in ['train', 'val', 'test']:
        cat_data = np.load(os.path.join(dataset_dir, category + '.npz'))
        data['x_' + category] = cat_data['x']  # B T N F speed flow
        data['y_' + category] = cat_data['y']

    train_max, val_max, test_max = data['x_train'].max(), data['x_val'].max(), data['x_test'].max()
    train_min, val_min, test_min = data['x_train'].min(), data['x_val'].min(), data['x_test'].min()
    max_list = np.asarray([train_max, val_max, test_max])
    min_list = np.asarray([train_min, val_min, test_min])

    data['scaler'] = StandardScaler(max=np.max(max_list), min=np.min(min_list), mean=data['x_train'].mean(), std=data['x_train'].std())
    scaler_tmp = data['scaler']
    for category in ['train', 'val', 'test']:
        data['x_' + category] = scaler_tmp.transform(data['x_' + category])

    data['train_loader'] = DataLoader(data['x_train'], data['y_train'], batch_size, days=days, begin=0)
    data['val_loader'] = DataLoader(data['x_val'], data['y_val'], valid_batch_size, days=days,
                                    begin=data['x_train'].shape[0])
    data['test_loader'] = DataLoader(data['x_test'], data['y_test'], test_batch_size, days=days,
                                     begin=data['x_train'].shape[0] + data['x_val'].shape[0])
    return data


def masked_mse(preds, labels, null_val=np.nan):
    if np.isnan(null_val):
        mask = ~torch.isnan(labels)
    else:
        mask = (labels != null_val)
    mask = mask.float()
    mask /= torch.mean((mask))
    mask = torch.where(torch.isnan(mask), torch.zeros_like(mask), mask)
    loss = (preds - labels) ** 2
    loss = loss * mask
    loss = torch.where(torch.isnan(loss), torch.zeros_like(loss), loss)
    return torch.mean(loss)


def masked_rmse(preds, labels):
    return torch.sqrt(torch.mean((preds - labels) ** 2))

def masked_mae(preds, labels):
    return torch.mean(torch.abs(preds - labels))

def masked_mape(preds, labels, threshold=10.):
    mask = labels > threshold
    if torch.sum(mask) != 0:
        mape = torch.mean(torch.abs(labels[mask] - preds[mask]) / labels[mask])
    return mape * 100


def metric(pred, real):
    outflow_mae = masked_mae(pred[:, 0:1, :, 0], real[:, 0:1, :, 0]).item()
    outflow_mape = masked_mape(pred[:, 0:1, :, 0], real[:, 0:1, :, 0], 10.0).item()
    outflow_rmse = masked_rmse(pred[:, 0:1, :, 0], real[:, 0:1, :, 0]).item()
    inflow_mae = masked_mae(pred[:, 1:2, :, 0], real[:, 1:2, :, 0]).item()
    inflow_mape = masked_mape(pred[:, 1:2, :, 0], real[:, 1:2, :, 0], 10.0).item()
    inflow_rmse = masked_rmse(pred[:, 1:2, :, 0], real[:, 1:2, :, 0]).item()
    return outflow_mae, outflow_mape, outflow_rmse, inflow_mae, inflow_mape, inflow_rmse
