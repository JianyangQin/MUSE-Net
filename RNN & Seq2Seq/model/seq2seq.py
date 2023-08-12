import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, input_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Encoder, self).__init__()

        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(input_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.dropout = nn.Dropout(dropout)

    def forward(self, src):
        # src = [src len, batch size, feat_dim]

        embedded = self.dropout(self.embedding(src))

        # embedded = [src len, batch size, emb dim]

        outputs, (hidden, cell) = self.rnn(embedded)

        # outputs = [src len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # outputs are always from the top hidden layer

        return outputs, hidden, cell


class Decoder(nn.Module):
    def __init__(self, output_dim, emb_dim, hid_dim, n_layers, dropout):
        super(Decoder, self).__init__()

        self.output_dim = output_dim
        self.hid_dim = hid_dim
        self.n_layers = n_layers

        self.embedding = nn.Linear(output_dim, emb_dim)

        self.rnn = nn.LSTM(emb_dim, hid_dim, n_layers, dropout=dropout)

        self.fc_out = nn.Linear(hid_dim, output_dim)

        self.dropout = nn.Dropout(dropout)

    def forward(self, input, hidden, cell):
        # input = [1, batch size, outpu_dim]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # n directions in the decoder will both always be 1, therefore:
        # hidden = [n layers, batch size, hid dim]
        # context = [n layers, batch size, hid dim]

        embedded = self.dropout(self.embedding(input))

        # embedded = [1, batch size, emb dim]

        output, (hidden, cell) = self.rnn(embedded, (hidden, cell))

        # output = [seq len, batch size, hid dim * n directions]
        # hidden = [n layers * n directions, batch size, hid dim]
        # cell = [n layers * n directions, batch size, hid dim]

        # seq len and n directions will always be 1 in the decoder, therefore:
        # output = [1, batch size, hid dim]
        # hidden = [n layers, batch size, hid dim]
        # cell = [n layers, batch size, hid dim]

        prediction = self.fc_out(output.squeeze(0))

        # prediction = [batch size, output dim]

        return prediction, hidden, cell


class Seq2Seq(nn.Module):
    def __init__(self, device, input_seq, output_seq, input_dim, emb_dim, hidden_dim, output_dim, num_of_layers):
        super(Seq2Seq, self).__init__()
        self.device = device
        self.output_seq = output_seq

        self.encoder = Encoder(input_dim, emb_dim, hidden_dim, num_of_layers, 0.1)
        self.fc_feature = nn.Linear(hidden_dim, output_dim)
        self.fc_seq = nn.Linear(input_seq, output_seq)
        self.decoder = Decoder(output_dim, emb_dim, hidden_dim, num_of_layers, 0.1)



    def forward(self, x):
        # x = [batch size， num_of_vertices, len_seq, feature_dim]

        # last hidden state of the encoder is used as the initial hidden state of the decoder
        x = x.permute(2, 1, 0, 3)
        len_seq, num_of_vertices, batch_size, feature_dim = x.shape

        y = []
        for n in range(num_of_vertices):
            xi = x[:, n, :, :]

            input, hidden, cell = self.encoder(xi)

            input = self.fc_seq(input.permute(1, 2, 0)).permute(2, 0, 1)
            input = self.fc_feature(input)

            outputs = []
            for t in range(self.output_seq):
                # insert input token embedding, previous hidden and previous cell states
                # receive output tensor (predictions) and new hidden and cell states
                output, hidden, cell = self.decoder(input, hidden, cell)

                # place predictions in a tensor holding predictions for each token
                outputs.append(output)

            outputs = torch.stack(outputs, dim=0)
            y.append(outputs)

        y = torch.stack(y, dim=1)
        y = y.permute(2, 1, 0, 3)

        return y