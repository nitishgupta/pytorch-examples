import re
import os
import gc
import sys
import math
import time
import pickle
import random
import unicodedata
import collections
import numpy as np
from tqdm import tqdm
import reader.utils as utils
from reader.Mention import Mention

start_word = "<s>"
end_word = "<eos>"

# person: 25, event: 10, organization: 5, location: 1


def AS(a):
    return a


class TrainingDataReader(object):
	def __init__(self, train_file, val_file, batch_size,
							 strict_context=True, pretrain_wordembed=False,
							 make_vocab=False):
		print("** Loading Training Reader **")
		self.start_word = start_word
		self.end_word = end_word
		self.unk_word = 'unk' # In tune with word2vec
		self.pretrain_wordembed = pretrain_wordembed
		self.strict_context = strict_context
		self.batch_size = batch_size

		print("* loading mentions *")
		self.tr_mens = utils.make_mentions_from_file(train_file)
		self.val_mens = utils.make_mentions_from_file(val_file)
		print("* mentions loaded *")

		if make_vocab:
			(w2idx, idx2w) = self.make_word_vocab(self.tr_mens)
			utils.save("reader/wordvocab.pkl", (w2idx, idx2w))

		(self.w2idx, self.idx2w) = utils.load("reader/wordvocab.pkl")
		self.numwords = len(self.idx2w)
		print("NumWords : {}".format(self.numwords))

		self.numlabels = 2


		self.numtrmens = len(self.tr_mens)
		print("[#] Training Mentions : {}".format(self.numtrmens))

		self.numvalmens = len(self.val_mens)
		print("[#] Validation Mentions : {}".format(self.numvalmens))

		self.tr_men_idx = 0
		self.tr_epochs = 0
		self.val_men_idx = 0
		self.val_epochs = 0

		print("[#] Batch Size: {}".format(self.batch_size))

		print("[#] LOADING COMPLETE:")

	#*******************      END __init__      *********************************

	def make_word_vocab(self, mens):
		w2id = {self.unk_word:0}
		id2w = [self.unk_word]
		for men in tqdm(mens):
			for w in men.sent_tokens:
				if w not in w2id:
					id2w.append(w)
					w2id[w] = len(id2w) - 1
		return (w2id, id2w)

	def reset_validation(self):
		self.val_men_idx = 0
		self.val_epochs = 0

	def get_mention(self, data_type):
		m = None
		if data_type == 0:
			m = self.tr_mens[self.tr_men_idx]
			self.tr_men_idx += 1
			if self.tr_men_idx == self.numtrmens:
				self.tr_men_idx = 0
				self.tr_epochs += 1
		elif data_type == 1:
			m = self.val_mens[self.val_men_idx]
			self.val_men_idx += 1
			if self.val_men_idx == self.numvalmens:
				self.val_men_idx = 0
				self.val_epochs += 1
		else:
			print("Wrong datatype")
			sys.exit()

		return m

	def _next_batch(self, data_type):
		''' Data : wikititle \t mid \t wid \t start \t end \t tokens \t labels
		start and end are inclusive
		'''
		# Sentence     = s1 ... m1 ... mN, ... sN.
		# Left Batch   = s1 ... m1 ... mN
		# Right Batch  = sN ... mN ... m1
		(left_batch, right_batch) = ([], [])

		# Person / Not person
		labels_batch = np.zeros([self.batch_size], dtype=int)

		while len(left_batch) < self.batch_size:
			m =  self.get_mention(data_type)
			batch_el = len(left_batch)

			start = m.start_token
			end = m.end_token

			# if person, id 1 is 1.0 o/w id=0 is 1.0
			if m.person == 1:
				labels_batch[batch_el] = 1.0

			# Left and Right context
			if self.strict_context:    # Strict Context
				left_tokens = m.sent_tokens[0:m.start_token]
				right_tokens = m.sent_tokens[m.end_token+1:][::-1]
			else:    # Context inclusive of mention surface
				left_tokens = m.sent_tokens[0:m.end_token+1]
				right_tokens = m.sent_tokens[m.start_token:][::-1]

			if not self.pretrain_wordembed:
				left_idxs = [self.convert_word2idx(word) for word in left_tokens]
				right_idxs = [self.convert_word2idx(word) for word in right_tokens]
			else:
				left_idxs = left_tokens
				right_idxs = right_tokens

			left_batch.append(left_idxs)
			right_batch.append(right_idxs)

		#end batch making
		return (left_batch, right_batch, labels_batch)
	#enddef

	def wordDropout(self, list_tokens, dropoutkeeprate):
		if dropoutkeeprate < 1.0:
				for i in range(0,len(list_tokens)):
					r = random.random()
					if r > dropoutkeeprate:
						list_tokens[i] = self.unk_word
		return list_tokens

	def print_test_batch(self, mention, wid_idxs, wid_cprobs):
		print("Surface : {}  WID : {}".format(mention.surface, mention.wid))
		print(mention.sent_tokens)
		# print("WIDS : ")
		# for (idx,cprob) in zip(wid_idxs, wid_cprobs):
		#   print("WID : {}  CPROB : {}".format(self.idx2knwid[idx], cprob))
		# print()

	def embed_batch(self, batch):
		''' Input is a padded batch of left or right contexts containing words
		Dimensions should be [B, padded_length]
		Output:
			Embed the word idxs using pretrain word embedding
		'''
		output_batch = []
		for sent in batch:
			word_embeddings = [self.get_vector(word) for word in sent]
			output_batch.append(word_embeddings)
		return output_batch

	def pad_batch(self, batch):
		if not self.pretrain_wordembed:
			padding = self.w2idx[self.unk_word]
		else:
			padding = self.unk_word
		lengths = [len(i) for i in batch]
		max_length = max(lengths)
		for i in range(0, len(batch)):
			batch[i].extend([padding]*(max_length - lengths[i]))
		return (batch, lengths)

	def _next_padded_batch(self, data_type):
		(left_batch, right_batch,
		 labels_batch) = self._next_batch(data_type=data_type)
		(left_batch, left_lengths) = self.pad_batch(left_batch)
		(right_batch, right_lengths) = self.pad_batch(right_batch)
		if self.pretrain_wordembed:
			left_batch = self.embed_batch(left_batch)
			right_batch = self.embed_batch(right_batch)
			#mention_batch = self.embed_mentions_batch(mention_batch)

		return (left_batch, left_lengths, right_batch, right_lengths, labels_batch)
	#enddef

	def convert_word2idx(self, word):
		if word in self.w2idx:
			return self.w2idx[word]
		else:
			return self.w2idx[self.unk_word]
	#enddef

	def next_train_batch(self):
		return self._next_padded_batch(data_type=0)

	def next_val_batch(self):
		return self._next_padded_batch(data_type=1)

	def next_test_batch(self):
		return self._next_padded_batch(data_type=2)

if __name__ == '__main__':
	sttime = time.time()
	batch_size = 10
	b = TrainingDataReader(train_file="data/train.data",
												 val_file="data/val.data", batch_size=batch_size,
												 strict_context=True, pretrain_wordembed=False,
												 make_vocab=False)

	stime = time.time()

	i = 0
	total_instances = 0
	labels_sum = np.array([0.0, 0.0])
	while b.tr_epochs < 1 and b.val_epochs < 1:
			(left_batch, left_lengths,
			 right_batch, right_lengths, labels_batch) = b.next_train_batch()
			total_instances += batch_size
			labels_sum += np.sum(labels_batch, 0)
			i += 1
	#endfor
	etime = time.time()
	t=etime-stime
	tt = etime - sttime
	print("NonPerson vs. Person : {}".format(labels_sum))
	print("Total Instances : {}".format(total_instances))
	print("Batching time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, t))
	print("Total time (in secs) to make %d batches of size %d : %7.4f seconds" % (i, batch_size, tt))
