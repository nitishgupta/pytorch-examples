import torch
import torch.nn as nn
from reader.trainreader import TrainingDataReader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq


class FF(nn.Module):
    def __init__(self, hidden_size, numlabels):
        super(FF, self).__init__()
        self.linear_l1 = nn.Linear(hidden_size, numlabels, False)
        self.linear_l2 = nn.Linear(numlabels, numlabels, False)
        self.relu = nn.ReLU()
        self.init_weights()

    def init_weights(self):
        self.linear_l1.weight.data.uniform_(-0.1, 0.1)
        self.linear_l2.weight.data.uniform_(-0.1, 0.1)

    def forward(self, x):
        out = self.linear_l1(x)
        out = self.relu(out)
        out = self.linear_l2(out)
        return out
