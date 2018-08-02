# https://github.com/spro/practical-pytorch

import torch
import torch.nn as nn
from torch.autograd import Variable

class RNN(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, n_layers=1, mode='lstm'):
        super(RNN, self).__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.n_layers = n_layers
        self.mode = mode

        self.encoder = nn.Embedding(input_size, hidden_size)
        if self.mode == 'gru':
            self.gru = nn.GRU(hidden_size, hidden_size, n_layers)
        else:
            self.lstm = nn.LSTM(hidden_size, hidden_size, num_layers=n_layers)
        self.decoder = nn.Linear(hidden_size, output_size)

    def forward(self, input, hidden):
        x = self.encoder(input.view(1, -1)).view(1, 1, -1)
        output, hidden = self.lstm(x, hidden)
        output = self.decoder(output.view(1, -1))
        return output, hidden

    def init_hidden(self):
        # (num_layers, minibatch_size, hidden_dim)
        shape = (self.n_layers, 1, self.hidden_size)
        if self.mode == 'gru':
            return Variable(torch.zeros(shape)).cuda()
        else:
            hx = Variable(torch.zeros(shape)).cuda()
            cx = Variable(torch.zeros(shape)).cuda()
            return (hx, cx)

