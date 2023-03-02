
from .data_utils import gen_batch
from .math_utils import evaluation
from os.path import join as pjoin
import numpy as np
import time
import torch


def multi_pred(model, seq, batch_size, n_his, n_pred, dynamic_batch=True):
    pred_list = []
    for i in gen_batch(seq, min(batch_size, len(seq)), dynamic_batch=dynamic_batch):
        # Note: use np.copy() to avoid the modification of source data.
        test_seq = np.copy(i[:, 0:n_his, :, :])
        step_list = []
        test_seq_th = torch.tensor(test_seq, dtype=torch.float32).cuda()
        pred = model(test_seq_th)
        pred = pred.data.cpu().numpy()
        pred_list.append(pred)
    #  pred_array -> [batch_size, n_route, C_0)
    pred_array = np.concatenate(pred_list, axis=0)
    return pred_array, pred_array.shape[0]


def model_inference(model, inputs, batch_size, n_his, n_pred, n):
    x_val, x_test, x_stats = inputs.get_data('val'), inputs.get_data('test'), inputs.get_stats()
    if n_his + n_pred > x_val[0].shape[1]:
        raise ValueError(f'ERROR: the value of n_pred "{n_pred}" exceeds the length limit.')
    y_val, len_val = multi_pred(model, x_val, batch_size, n_his, n_pred)
    evl_val_outflow, evl_val_inflow = evaluation(x_val[0:len_val, n_his:n_pred + n_his, : n, :], y_val[:, :, : n], x_stats)
    # update the metric on test set, if model's performance got improved on the validation.
    y_pred, len_pred = multi_pred(model, x_test, batch_size, n_his, n_pred)
    evl_pred_outflow, evl_pred_inflow = evaluation(x_test[0:len_pred, n_his:n_pred + n_his, : n, :], y_pred[:, :, : n], x_stats)
    # min_val = evl_pred
    return evl_val_outflow, evl_val_inflow, evl_pred_outflow, evl_pred_inflow


