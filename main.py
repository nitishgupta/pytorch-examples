import pdb
import time
import math
import argparse
from tqdm import tqdm


import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable

import reader.utils as utils
from reader.trainreader import TrainingDataReader


import model.utils as utils
from model.model import Model



parser = argparse.ArgumentParser(description='PyTorch RNN/LSTM Classification Model')
parser.add_argument('--trdata', type=str, default='./data/train.data',
                    help='location of the data corpus')
parser.add_argument('--valdata', type=str, default='./data/val.data',
                    help='location of the data corpus')
parser.add_argument('--testdata', type=str, default='./data/test.data',
                    help='location of the data corpus')
parser.add_argument('--emsize', type=int, default=100,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=100,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=1,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=5e-3,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=5,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=100, metavar='N',
                    help='batch size')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='model.pt',
                    help='path to save the final model')
args = parser.parse_args()

args = parser.parse_args()

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)


""" Reader, model and optimizer should be in __init__ for Train class """

trreader = TrainingDataReader(train_file=args.trdata, val_file=args.valdata,
							  batch_size=args.batch_size, strict_context=True)
model = Model(vocab_size=trreader.numwords, embed_size=args.emsize,
			  hidden_size=args.nhid, num_layers=args.nlayers,
			  numlabels=trreader.numlabels)
model.cuda()
optimizer = optim.Adam(model.parameters())
# COMMENT(nitish) : Decide where to go
criterion = nn.CrossEntropyLoss()
""" In __init__ up till here """

def optstep(loss):
	""" Function in Train class
	Optimizer should be a self of Train class, so only loss needed
	Could also include gradient clipping by using self.model.parameters()
	"""
	optimizer.zero_grad()
	loss.backward()
	optimizer.step()


def train():
	""" train function in Train class
	Figure if model is to be loaded. Load model and opt.
	loop over batches, get output, calc. loss, make optstep
	save model at intervals and possibly call validation
	"""

	print(model.ffnet.linear_l2.weight)
	savepath = "m.save"
	steps = utils.load_checkpoint(savepath, model, optimizer)
	print(model.ffnet.linear_l2.weight)

	epochs = trreader.tr_epochs
	steps = 0
	while trreader.tr_epochs < 1:
		steps += 1
		(leftidxs, leftlens, rightidxs, rightlens, labels) = trreader.next_train_batch()
		x = Variable(torch.LongTensor(leftidxs)).cuda()
		lens = Variable(torch.LongTensor(leftlens)).cuda()
		labels = Variable(torch.LongTensor(labels)).cuda()
		out = model(x, lens)
		loss = model.loss(output=out, target=labels, criterion=criterion)
		optstep(loss)

		if steps % 100 == 0:
			print(loss.data[0])

		if epochs != trreader.tr_epochs:
			epochs = trreader.tr_epochs
			print(epochs)

		optimizer.zero_grad()

	utils.save_checkpoint(model, optimizer, steps, savepath)

if __name__=="__main__":
	train()
