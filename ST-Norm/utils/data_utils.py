import numpy as np
import pandas as pd


class Dataset(object):
    def __init__(self, data, stats):
        self.__data = data
        self.max = stats['max']
        self.min = stats['min']

    def get_data(self, type):
        return self.__data[type]

    def get_stats(self):
        return {'max': self.max, 'min': self.min}

    def get_len(self, type):
        return len(self.__data[type])


def seq_gen(len_seq, data_seq, offset, n_frame, n_route, day_slot, C_0=1):
    n_slot = day_slot

    tmp_seq = np.zeros((len_seq * n_slot, n_frame, n_route, C_0))
    for i in range(len_seq):
        for j in range(n_slot):
            end = (i + offset) * day_slot + j + 1
            sta = end - n_frame
            if sta >= 0:
                tmp_seq[i * n_slot + j, :, :, :] = np.reshape(data_seq[sta:end, :], [n_frame, n_route, C_0])

    return tmp_seq


def gen_batch(inputs, batch_size, dynamic_batch=False, shuffle=False, period=None):
    len_inputs = len(inputs)
    if shuffle:
        idx = np.arange(len_inputs)
        np.random.shuffle(idx)

    for start_idx in range(0, len_inputs, batch_size):
        end_idx = start_idx + batch_size
        if end_idx > len_inputs:
            if dynamic_batch:
                end_idx = len_inputs
            else:
                break
        if shuffle:
            slide = idx[start_idx:end_idx]
        else:
            slide = slice(start_idx, end_idx)

        yield inputs[slide]

from utils.read_data import load_data

def traffic_data_gen(dataset, args):
    (x_train, x_val, x_test, max, min) = load_data(dataset, args)
    x_stats = {'max': max, 'min': min}
    x_data = {'train': x_train, 'val': x_val, 'test': x_test}
    dataset = Dataset(x_data, x_stats)
    return dataset
