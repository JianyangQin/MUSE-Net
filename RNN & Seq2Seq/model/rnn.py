import torch
import torch.nn as nn

class RNN(nn.Module):
    def __init__(self, device, in_seq, out_seq, input_dim, hidden_dim, num_of_layers):
        super(RNN, self).__init__()
        self.device = device
        self.hidden_dim = hidden_dim
        self.num_of_layers = num_of_layers
        self.rnn = nn.RNN(input_size=input_dim,
                          hidden_size=hidden_dim,
                          num_layers=num_of_layers,
                          batch_first=True)
        self.fc_feature = nn.Linear(hidden_dim, input_dim)
        self.fc_seq = nn.Linear(in_seq, out_seq)

    def forward(self, x):
        batch_size, num_of_vertices, len_seq, feature_dim = x.shape
        y = []
        for i in range(num_of_vertices):
            xi = x[:, i, :, :]
            h_0 = torch.zeros(self.num_of_layers, batch_size, self.hidden_dim).to(self.device)
            xi, h_0 = self.rnn(xi, h_0)
            xi = self.fc_seq(xi.permute(0, 2, 1)).permute(0, 2, 1)
            xi = self.fc_feature(xi)
            y.append(xi)
        y = torch.stack(y, dim=1)
        return y


