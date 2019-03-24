import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import numpy
np = numpy

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU

# Use the GPU if you have one
if torch.cuda.is_available():
	print("Using the GPU")
	device = torch.device("cuda")
else:
	print("WARNING: You are about to run on cpu, and this will likely run out \
		of memory. \n You can try setting batch_size=1 to reduce memory usage")
	device = torch.device("cpu")


###############################################################################
#
# DATA LOADING & PROCESSING
#
###############################################################################

# HELPER FUNCTIONS
def _read_words(filename):
    with open(filename, "r") as f:
      return f.read().replace("\n", "<eos>").split()

def _build_vocab(filename):
    data = _read_words(filename)

    counter = collections.Counter(data)
    count_pairs = sorted(counter.items(), key=lambda x: (-x[1], x[0]))

    words, _ = list(zip(*count_pairs))
    word_to_id = dict(zip(words, range(len(words))))
    id_to_word = dict((v, k) for k, v in word_to_id.items())

    return word_to_id, id_to_word

def _file_to_word_ids(filename, word_to_id):
    data = _read_words(filename)
    return [word_to_id[word] for word in data if word in word_to_id]

# Processes the raw data from text files
def ptb_raw_data(data_path=None, prefix="ptb"):
    train_path = os.path.join(data_path, prefix + ".train.txt")
    valid_path = os.path.join(data_path, prefix + ".valid.txt")
    test_path = os.path.join(data_path, prefix + ".test.txt")

    word_to_id, id_2_word = _build_vocab(train_path)
    train_data = _file_to_word_ids(train_path, word_to_id)
    valid_data = _file_to_word_ids(valid_path, word_to_id)
    test_data = _file_to_word_ids(test_path, word_to_id)
    return train_data, valid_data, test_data, word_to_id, id_2_word


###############################################################################
#
# GENERATING SEQUENCES
#
###############################################################################

def generate_sequence(num_samples, model_type, emb_size, hidden_size, seq_len, batch_size, num_layers, dp_keep_prob, vocab_size, path):
	if model_type=='RNN':
		model = RNN(emb_size=emb_size, hidden_size=hidden_size,
				seq_len=seq_len, batch_size=batch_size,
				vocab_size=vocab_size, num_layers=num_layers,
				dp_keep_prob=dp_keep_prob)
	else:
		model = GRU(emb_size=emb_size, hidden_size=hidden_size,
				seq_len=seq_len, batch_size=batch_size,
				vocab_size=vocab_size, num_layers=num_layers,
				dp_keep_prob=dp_keep_prob)

	model.load_state_dict(torch.load(path))
	model = model.to(device)
	hidden = nn.Parameter(torch.zeros(num_layers, num_samples, hidden_size)).to(device)
	input = torch.ones(10000)*1/1000
	input = torch.multinomial(input, num_samples).to(device)
	output = model.generate(input, hidden, seq_len)

	fid= open(model_type + '_generated_sequences' +'.txt','w')
	for i in range(num_samples):
		for j in range(seq_len):
			fid.write(id_2_word.get(output[j,i].item())+' ')
		fid.write('\n')
	fid.close()

print('GENERATING SEQUENCES')
print()
raw_data = ptb_raw_data(data_path='data')
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('vocabulary size: {}'.format(vocab_size))

# For RNN
rnn_path = 'Question_4.3/RNN_DP_0.45/best_params.pt'

generate_sequence(num_samples=500, model_type='RNN', emb_size=200, hidden_size=1500, seq_len=35, batch_size=32, num_layers=2, dp_keep_prob=0.45, vocab_size=vocab_size, path=rnn_path)


# # For GRU
# gru_path = 'Question_4.3/____________/best_params.pt'
# generate_sequence((num_samples=100, model_type='GRU', emb_size=200, hidden_size=1500, seq_len=35, batch_size=32, num_layers=2, dp_keep_prob=0.35, vocab_size=vocab_size, path=gru_path)
