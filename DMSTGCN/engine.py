import torch.optim as optim
from model import *
import util


class trainer():
    def __init__(self, scaler, in_dim, in_len, seq_length, num_nodes, nhid, dropout, normalization, lrate, wdecay, device, days=288,
                 dims=40, order=2):
        self.model = DMSTGCN_2flow(device, num_nodes, dropout, in_len=in_len, out_dim=seq_length, residual_channels=nhid,
                                   dilation_channels=nhid, end_channels=nhid * 16, days=days, dims=dims, order=order,
                                   in_dim=in_dim, normalization=normalization)

        self.model.to(device)
        self.optimizer = optim.Adam(self.model.parameters(), lr=lrate, weight_decay=wdecay)
        self.scaeduler = optim.lr_scheduler.ReduceLROnPlateau(self.optimizer, factor=.3, patience=10, threshold=1e-3,
                                                              min_lr=1e-5, verbose=True)
        self.loss = util.masked_mae
        self.scaler = scaler
        self.clip = 5

    def train(self, input, real_val, ind):
        self.model.train()
        self.optimizer.zero_grad()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input, ind)
        output = output.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)
        loss = self.loss(predict, real_val)
        loss.backward()
        if self.clip is not None:
            torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.clip)
        self.optimizer.step()
        outflow_mae = util.masked_mae(predict[:, 0:1, :, 0], real_val[:, 0:1, :, 0]).item()
        outflow_mape = util.masked_mape(predict[:, 0:1, :, 0], real_val[:, 0:1, :, 0], 10.0).item()
        outflow_rmse = util.masked_rmse(predict[:, 0:1, :, 0], real_val[:, 0:1, :, 0]).item()
        inflow_mae = util.masked_mae(predict[:, 1:2, :, 0], real_val[:, 1:2, :, 0]).item()
        inflow_mape = util.masked_mape(predict[:, 1:2, :, 0], real_val[:, 1:2, :, 0], 10.0).item()
        inflow_rmse = util.masked_rmse(predict[:, 1:2, :, 0], real_val[:, 1:2, :, 0]).item()
        return outflow_mae, outflow_mape, outflow_rmse, inflow_mae, inflow_mape, inflow_rmse

    def eval(self, input, real_val, ind):
        self.model.eval()
        input = nn.functional.pad(input, (1, 0, 0, 0))
        output = self.model(input, ind)
        output = output.transpose(1, 3)
        predict = self.scaler.inverse_transform(output)
        outflow_mae = util.masked_mae(predict[:, 0:1, :, 0], real_val[:, 0:1, :, 0]).item()
        outflow_mape = util.masked_mape(predict[:, 0:1, :, 0], real_val[:, 0:1, :, 0], 10.0).item()
        outflow_rmse = util.masked_rmse(predict[:, 0:1, :, 0], real_val[:, 0:1, :, 0]).item()
        inflow_mae = util.masked_mae(predict[:, 1:2, :, 0], real_val[:, 1:2, :, 0]).item()
        inflow_mape = util.masked_mape(predict[:, 1:2, :, 0], real_val[:, 1:2, :, 0], 10.0).item()
        inflow_rmse = util.masked_rmse(predict[:, 1:2, :, 0], real_val[:, 1:2, :, 0]).item()
        return outflow_mae, outflow_mape, outflow_rmse, inflow_mae, inflow_mape, inflow_rmse
