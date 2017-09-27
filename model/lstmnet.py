import torch
import torch.nn as nn
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq


class RNNModel(nn.Module):
    def __init__(self, vocab_size, embed_size, hidden_size, num_layers):
        super(RNNModel, self).__init__()
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers,
                            batch_first=True)

    def forward(self, x, lens, wordembedmat):
        lstm_output = self._getLastLSTMOutput(self.lstm, x, wordembedmat, lens)
        # lstm_output = self._getlastlstm(self.lstm, x, wordembedmat, lens)
        return lstm_output

        def _getlastlstm(self, lstm, x, wordembed, lens):
            bs = x.size()[0]
            h0 = Variable(torch.zeros(1, bs, lstm.input_size)).cuda()
            c0 = Variable(torch.zeros(1, bs, lstm.input_size)).cuda()
            x = wordembed(x)
            out, h = lstm(x, (h0, c0))
            # out = out[:, -1, :]
            out = h[0][0]
            return out

        def _getLastLSTMOutput(self, lstm, x, wordembed, lens):
            x = wordembed(x)
            bs = x.size()[0]
            h0 = Variable(torch.zeros(1, bs, lstm.input_size)).cuda()
            c0 = Variable(torch.zeros(1, bs, lstm.input_size)).cuda()

            # print("X : {} lens : {}".format(x, lens))
            sortedlens, sortedidxs = torch.sort(lens, dim=0, descending=True)

            print(sortedidxs)
            x = x[sortedidxs.data]

            _, originalidxs = torch.sort(sortedidxs, dim=0)
            print("Original Idxs")
            print(originalidxs)

            # Embed word ids to vectors
            packed_x = packseq(x, list(sortedlens.data), batch_first=True)

            # Forward propagate RNN
            out, (h, c) = lstm(packed_x, (h0, c0))
            h = h.squeeze(0)
            out, _ = padseq(out, batch_first=True)
            out = out[originalidxs.data]
            print(out)

            # idx = (sortedlens - 1).view(-1, 1).expand(bs, self.lstm.hidden_size).unsqueeze(1)
            # decoded_sorted = out.gather(1, idx).squeeze()
            # h == decoded_sorted

            oridx = originalidxs.view(-1, 1).expand(bs, out.size(-1))
            lstm_output_h = h.gather(0, oridx)

            print(lstm_output_h)
            return lstm_output_h




            sortedlens, sortedidxs = torch.sort(lens, dim=0, descending=True)
            x = x[sortedidxs.data]

            # print("X sorted : {} slens : {} sidxs : {}".format(
            # x, sortedlens, sortedidxs))

            # Embed word ids to vectors
            x = wordembed(x)
            packed_x = packseq(x, list(sortedlens.data), batch_first=True)

            # Forward propagate RNN
            out, h = lstm(packed_x, (h0, c0))
            out, _ = padseq(out, batch_first=True)
            # print("Out : {}".format(out))

            idx = (sortedlens - 1).view(-1, 1).expand(
                bs, self.lstm.hidden_size).unsqueeze(1)
            decoded_sorted = out.gather(1, idx).squeeze()
            bs = x.size()[0]
            odx = sortedidxs.view(-1, 1).expand(bs, out.size(-1))
            lstm_output = decoded_sorted.gather(0, odx)
            return lstm_output
