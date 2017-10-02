import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq

from reader.trainreader import TrainingDataReader

from model import lstmnet


class Model(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers,
                 numlabels):
        super(Model, self).__init__()
        self.wordembed = nn.Embedding(vocab_size, embed_size)
        self.wordembed.weight.data.uniform_(-0.1, 0.1)
        print(self.wordembed.weight.size())


        self.llstm = lstmnet.RNNModel(vocab_size, embed_size, hidden_size, num_layers)
        self.rlstm = lstmnet.RNNModel(vocab_size, embed_size, hidden_size, num_layers)
        print("LLSTM : {}".format(self.llstm.state_dict().keys()))
        print("RLSTM : {}".format(self.rlstm.state_dict().keys()))
        self.ffnet = FF(hidden_size, numlabels)
        print("FFNET : {}".format(self.ffnet.state_dict().keys()))

    def forward(self, x, lens):
        llstmout = self.llstm(x, lens, self.wordembed)
        out = self.ffnet(llstmout)
        return out

    def loss(self, output, target, criterion):
        # output : [B, 2], target : [B]
        loss = criterion(output, target)
        return loss
