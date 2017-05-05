import os
import torch
import torch.nn as nn
from collections import defaultdict
from reader.trainreader import TrainingDataReader
from torch.autograd import Variable
from torch.nn.utils.rnn import pack_padded_sequence as packseq
from torch.nn.utils.rnn import pad_packed_sequence as padseq


class Model(nn.Module):
	def __init__(self):
		super(Model, self).__init__()

		self.linear_l2 = nn.Linear(3,2, False)
		self.linear_l1 = nn.Linear(2,3, False)
		


	def forward(self, x):
		y = x
		y = self.linear_l1(x)
		y = self.linear_l2(y)
		return y


def load(m, path):
	# Current parameters
	# All are currently initialized
	state = m.state_dict()

	# Loaded state. This could have more parameters than current model, or missing some
	loaded_state = torch.load(path)

	# This set has the intersection of current and loaded
	keystobeupdated = set(state.keys()).intersection(set(loaded_state.keys()))
	# For params in current and saved, use saved
	for k in keystobeupdated:
		state[k] = loaded_state[k]

	m.load_state_dict(state)

def save(m, path):
	torch.save(m.state_dict(), path)

def save_optim(o, path):
	torch.save(o.state_dict(), path)

def load_optim(o, path):
	o.load_state_dict(torch.load(path))
	o.state = defaultdict(dict, o.state)


def train(m, o, c):
	x = Variable(torch.randn(1,2))
	y = m(x)

	loss = c(y, x)
	o.zero_grad()
	loss.backward()
	o.step()

	#print("x : {}".format(x.data))
	#print("y : {}".format(y.data))
	#print("loss : {}".format(loss.data))


m = Model()
o = torch.optim.Adam(m.parameters())
c = nn.MSELoss()

print(o.state_dict())
if os.path.exists("m.save"):
	load(m, "m.save")
if os.path.exists("o.save"):
	load_optim(o, "o.save")

print("PRE TRAINING")
print(m.linear_l1.weight)
print(o.state_dict())
print("\n")

for i in range(0, 5):
	train(m, o, c)

print("POST TRAINING")
print(m.linear_l1.weight)
print(o.state_dict())
save(m, "m.save")
save_optim(o, "o.save")

print(list(m.parameters()))