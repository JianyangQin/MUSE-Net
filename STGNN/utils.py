import numpy as np
import pandas as pd

# log string
def log_string(log, string):
    log.write(string + '\n')
    log.flush()
    print(string)

# metric
def metric(pred, label, threshold=10.0):
    with np.errstate(divide = 'ignore', invalid = 'ignore'):
        mae = np.abs(np.subtract(pred, label)).astype(np.float32)

        rmse = np.square(mae)

        mae = np.mean(mae)
        rmse = np.sqrt(np.mean(rmse))

        mask = label > threshold
        if np.sum(mask) != 0:
            mape = np.mean(np.abs(label[mask] - pred[mask]).astype('float64') / label[mask])
    return mae, rmse, mape