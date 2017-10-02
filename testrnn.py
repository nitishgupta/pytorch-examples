import sys
import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq


class RNNModel(nn.Module):
    def __init__(self, embed_size, hidden_size, num_layers=1):
        super(RNNModel, self).__init__()
        self.device_id = None

        self.input = torch.randn(3, 3, embed_size)
        lens = [2, 1, 3]
        self.lens = torch.LongTensor(lens)
        self.input = self._cuda(Variable(self.input))
        self.lens = self._cuda(Variable(self.lens))

        print(self.input)
        print(self.lens)

        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers, batch_first=True)

    def _cuda(self, m):
        if self.device_id is not None:
            return m.cuda(self.device_id)
        return m

    def forward(self):
        print("****** Without padding *********")
        self._getlastlstm()
        print("****** WITH PADDING ************")
        self._getLastLSTMOutput()

        return None

    def _getlastlstm(self):
        x = self.input
        lstm = self.lstm
        bs = x.size()[0]
        h0 = self._cuda(Variable(torch.zeros(1, bs, lstm.hidden_size)))
        c0 = self._cuda(Variable(torch.zeros(1, bs, lstm.hidden_size)))

        print(x)
        print(h0)
        print(h0)

        out, (h, c) = lstm(x, (h0, c0))
        print("Out")
        print(out)
        print("H")
        print(h)
        return out

    def _getLastLSTMOutput(self):
        x = self.input
        lstm = self.lstm
        lens = self.lens
        bs = x.size()[0]
        h0 = self._cuda(Variable(torch.zeros(1, bs, lstm.hidden_size)))
        c0 = self._cuda(Variable(torch.zeros(1, bs, lstm.hidden_size)))

        sortedlens, sortedidxs = torch.sort(lens, dim=0, descending=True)
        x = x[sortedidxs.data]

        packed_x = packseq(x, list(sortedlens.data), batch_first=True)

        # Forward propagate RNN
        out, (h, c) = lstm(packed_x, (h0, c0))
        h = h.squeeze(0)
        out, _ = padseq(out, batch_first=True)
        # out = out[originalidxs.data]
        print("Out: {}".format(out.data))
        print("H : {}".format(h.data))
        hlast = self._cuda(Variable(torch.FloatTensor(bs, out.size(2))))
        outlast = self._cuda(Variable(torch.FloatTensor(bs,
                                      out.size(1), out.size(2))))

        soridx2d = sortedidxs.unsqueeze(1).expand(bs, out.size(2))
        soridx3d = sortedidxs.unsqueeze(1).unsqueeze(2).expand(
            bs, out.size(1), out.size(2))
        hlast = hlast.scatter_(0, soridx2d, h)
        outlast = outlast.scatter_(0, soridx3d, out)

        print("Out unsorted : {}".format(outlast))
        print("h unsorted : {}".format(hlast))
        return hlast

if __name__ == '__main__':
    rnnmodel = RNNModel(embed_size=4, hidden_size=3, num_layers=1)
    # rnnmodel.cuda(None)
    rnnmodel.forward()
