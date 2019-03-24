#!/bin/python
# coding: utf-8

# Code outline/scaffold for
# ASSIGNMENT 2: RNNs, Attention, and Optimization
# By Tegan Maharaj, David Krueger, and Chin-Wei Huang
# IFT6135 at University of Montreal
# Winter 2019
#
# based on code from:
#    https://github.com/deeplearningathome/pytorch-language-model/blob/master/reader.py
#    https://github.com/ceshine/examples/blob/master/word_language_model/main.py
#    https://github.com/teganmaharaj/zoneout/blob/master/zoneout_word_ptb.py
#    https://github.com/harvardnlp/annotated-transformer

# GENERAL INSTRUCTIONS:
#    - ! IMPORTANT!
#      Unless we're otherwise notified we will run exactly this code, importing
#      your models from models.py to test them. If you find it necessary to
#      modify or replace this script (e.g. if you are using TensorFlow), you
#      must justify this decision in your report, and contact the TAs as soon as
#      possible to let them know. You are free to modify/add to this script for
#      your own purposes (e.g. monitoring, plotting, further hyperparameter
#      tuning than what is required), but remember that unless we're otherwise
#      notified we will run this code as it is given to you, NOT with your
#      modifications.
#    - We encourage you to read and understand this code; there are some notes
#      and comments to help you.
#    - Typically, all of your code to submit should be written in models.py;
#      see further instructions at the top of that file / in TODOs.
#          - RNN recurrent unit
#          - GRU recurrent unit
#          - Multi-head attention for the Transformer
#    - Other than this file and models.py, you will probably also write two
#      scripts. Include these and any other code you write in your git repo for
#      submission:
#          - Plotting (learning curves, loss w.r.t. time, gradients w.r.t. hiddens)
#          - Loading and running a saved model (computing gradients w.r.t. hiddens,
#            and for sampling from the model)

# PROBLEM-SPECIFIC INSTRUCTIONS:
#    - For Problems 1-3, paste the code for the RNN, GRU, and Multi-Head attention
#      respectively in your report, in a monospace font.
#    - For Problem 4.1 (model comparison), the hyperparameter settings you should run are as follows:
#          --model=RNN --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
#          --model=GRU --optimizer=SGD_LR_SCHEDULE --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35 --save_best
#          --model=TRANSFORMER --optimizer=SGD_LR_SCHEDULE --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=0.9 --save_best
#    - In those experiments, you should expect to see approximately the following
#    - In those experiments, you should expect to see approximately the following
#      perplexities:
#                  RNN: train:  120  val: 157
#                  GRU: train:   65  val: 104
#          TRANSFORMER:  train:  67  val: 146
#    - For Problem 4.2 (exploration of optimizers), you will make use of the
#      experiments from 4.1, and should additionally run the following experiments:
#          --model=RNN --optimizer=SGD --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=GRU --optimizer=SGD --initial_lr=10 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=TRANSFORMER --optimizer=SGD --initial_lr=20 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=6 --dp_keep_prob=.9
#          --model=RNN --optimizer=SGD_LR_SCHEDULE --initial_lr=1 --batch_size=20 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=0.35
#          --model=GRU --optimizer=ADAM --initial_lr=0.0001 --batch_size=20 --seq_len=35 --hidden_size=1500 --num_layers=2 --dp_keep_prob=0.35
#          --model=TRANSFORMER --optimizer=ADAM --initial_lr=0.001 --batch_size=128 --seq_len=35 --hidden_size=512 --num_layers=2 --dp_keep_prob=.9
#    - For Problem 4.3 (exloration of hyperparameters), do your best to get
#      better validation perplexities than the settings given for 4.1. You may
#      try any combination of the hyperparameters included as arguments in this
#      script's ArgumentParser, but do not implement any additional
#      regularizers/features. You may (and will probably want to) run a lot of
#      different things for just 1-5 epochs when you are trying things out, but
#      you must report at least 3 experiments on each architecture that have run
#      for at least 40 epochs.
#    - For Problem 5, perform all computations / plots based on saved models
#      from Problem 4.1. NOTE this means you don't have to save the models for
#      your exploration, which can make things go faster. (Of course
#      you can still save them if you like; just add the flag --save_best).
#    - For Problem 5.1, you can modify the loss computation in this script
#      (search for "LOSS COMPUTATION" to find the appropriate line. Remember to
#      submit your code.
#    - For Problem 5.3, you must implement the generate method of the RNN and
#      GRU.  Implementing this method is not considered part of problems 1/2
#      respectively, and will be graded as part of Problem 5.3


import argparse
import time
import collections
import os
import sys
import torch
import torch.nn
from torch.autograd import Variable
import torch.nn as nn
import matplotlib.pyplot as plt
import numpy
np = numpy

# NOTE ==============================================
# This is where your models are imported
from models import RNN, GRU
from models import make_model as TRANSFORMER


##############################################################################
#
# ARG PARSING AND EXPERIMENT SETUP
#
##############################################################################

parser = argparse.ArgumentParser(description='PyTorch Penn Treebank Language Modeling')

# Arguments you may need to set to run different experiments in 4.1 & 4.2.
parser.add_argument('--data', type=str, default='data',
                    help='location of the data corpus. We suggest you change the default\
                    here, rather than passing as an argument, to avoid long file paths.')
parser.add_argument('--model', type=str, default='GRU',
                    help='type of recurrent net (RNN, GRU, TRANSFORMER)')
parser.add_argument('--optimizer', type=str, default='SGD_LR_SCHEDULE',
                    help='optimization algo to use; SGD, SGD_LR_SCHEDULE, ADAM')
parser.add_argument('--seq_len', type=int, default=35,
                    help='number of timesteps over which BPTT is performed')
parser.add_argument('--batch_size', type=int, default=20,
                    help='size of one minibatch')
parser.add_argument('--initial_lr', type=float, default=20.0,
                    help='initial learning rate')
parser.add_argument('--hidden_size', type=int, default=200,
                    help='size of hidden layers. IMPORTANT: for the transformer\
                    this must be a multiple of 16.')
parser.add_argument('--save_best', action='store_true',
                    help='save the model for the best validation performance')
parser.add_argument('--num_layers', type=int, default=2,
                    help='number of LSTM layers')

# Other hyperparameters you may want to tune in your exploration
parser.add_argument('--emb_size', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--num_epochs', type=int, default=40,
                    help='number of epochs to stop after')
parser.add_argument('--dp_keep_prob', type=float, default=0.35,
                    help='dropout *keep* probability. drop_prob = 1-dp_keep_prob \
                    (dp_keep_prob=1 means no dropout)')

# Arguments that you may want to make use of / implement more code for
parser.add_argument('--debug', action='store_true')
parser.add_argument('--save_dir', type=str, default='',
                    help='path to save the experimental config, logs, model \
                    This is automatically generated based on the command line \
                    arguments you pass and only needs to be set if you want a \
                    custom dir name')
parser.add_argument('--evaluate', action='store_true',
                    help="use this flag to run on the test set. Only do this \
                    ONCE for each model setting, and only after you've \
                    completed ALL hyperparameter tuning on the validation set.\
                    Note we are not requiring you to do this.")
# DO NOT CHANGE THIS (setting the random seed makes experiments deterministic,
# which helps for reproducibility)
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')

args = parser.parse_args()
argsdict = args.__dict__
argsdict['code_file'] = sys.argv[0]

# Use the model, optimizer, and the flags passed to the script to make the
# name for the experimental dir
print("\n########## Setting Up Experiment ######################")
flags = [flag.lstrip('--') for flag in sys.argv[1:]]
experiment_path = os.path.join(args.save_dir+'_'.join([argsdict['model'],
                                                       argsdict['optimizer']]
                                                      + flags))

# Increment a counter so that previous results with the same args will not
# be overwritten. Comment out the next four lines if you only want to keep
# the most recent results.
i = 0
while os.path.exists(experiment_path + "_" + str(i)):
    i += 1
experiment_path = experiment_path + "_" + str(i)

# Creates an experimental directory and dumps all the args to a text file
os.mkdir(experiment_path)
print ("\nPutting log in %s"%experiment_path)
argsdict['save_dir'] = experiment_path
with open (os.path.join(experiment_path,'exp_config.txt'), 'w') as f:
    for key in sorted(argsdict):
        f.write(key+'    '+str(argsdict[key])+'\n')

# Set the random seed manually for reproducibility.
torch.manual_seed(args.seed)

#Use the GPU if you have one
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

# Yields minibatches of data
def ptb_iterator(raw_data, batch_size, num_steps):
    raw_data = np.array(raw_data, dtype=np.int32)

    data_len = len(raw_data)
    batch_len = data_len // batch_size
    data = np.zeros([batch_size, batch_len], dtype=np.int32)
    for i in range(batch_size):
        data[i] = raw_data[batch_len * i:batch_len * (i + 1)]

    epoch_size = (batch_len - 1) // num_steps

    if epoch_size == 0:
        raise ValueError("epoch_size == 0, decrease batch_size or num_steps")

    for i in range(epoch_size):
        x = data[:, i*num_steps:(i+1)*num_steps]
        y = data[:, i*num_steps+1:(i+1)*num_steps+1]
        yield (x, y)


class Batch:
    "Data processing for the transformer. This class adds a mask to the data."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."

        def subsequent_mask(size):
            """ helper function for creating the masks. """
            attn_shape = (1, size, size)
            subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
            return torch.from_numpy(subsequent_mask) == 0

        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


# LOAD DATA
print('Loading data from '+args.data)
raw_data = ptb_raw_data(data_path=args.data)
train_data, valid_data, test_data, word_to_id, id_2_word = raw_data
vocab_size = len(word_to_id)
print('  vocabulary size: {}'.format(vocab_size))


###############################################################################
#
# MODEL SETUP
#
###############################################################################

# LOSS FUNCTION
loss_fn = nn.CrossEntropyLoss()
if args.optimizer == 'ADAM':
    optimizer = torch.optim.Adam(model.parameters(), lr=args.initial_lr)

# LEARNING RATE SCHEDULE
lr = args.initial_lr
lr_decay_base = 1 / 1.15
m_flat_lr = 14.0 # we will not touch lr for the first m_flat_lr epochs


###############################################################################
#
# DEFINE COMPUTATIONS FOR PROCESSING ONE EPOCH
#
###############################################################################

def repackage_hidden(h):
    """
    Wraps hidden states in new Tensors, to detach them from their history.
    This prevents Pytorch from trying to backpropagate into previous input
    sequences when we use the final hidden states from one mini-batch as the
    initial hidden states for the next mini-batch.
    Using the final hidden states in this way makes sense when the elements of
    the mini-batches are actually successive subsequences in a set of longer sequences.
    This is the case with the way we've processed the Penn Treebank dataset.
    """
    if isinstance(h, Variable):
        return h.detach_()
    else:
        return tuple(repackage_hidden(v) for v in h)


def run_epoch(model, data, model_type):
    """
    One epoch of training/validation (depending on flag is_train).
    """
    model.eval()

    if  model_type != 'TRANSFORMER':
        hidden = model.init_hidden()
        hidden = hidden.to(device)

    if  model_type == 'TRANSFORMER':
        model.batch_size = 128
        model.seq_len = 25

    losses = []
    # LOOP THROUGH MINIBATCHES
    for step, (x, y) in enumerate(ptb_iterator(data, model.batch_size, model.seq_len)):
        if model_type != 'TRANSFORMER':
            hidden = model.init_hidden()
            hidden = hidden.to(device)

        if model_type == 'TRANSFORMER':
            batch = Batch(torch.from_numpy(x).long())
            model.zero_grad()
            outputs = model.forward(batch.data, batch.mask).transpose(1,0)
            # print ("outputs.shape", outputs.shape)
        else:
            inputs = torch.from_numpy(x.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
            model.zero_grad()
            hidden = repackage_hidden(hidden)
            outputs, hidden = model(inputs, hidden)
            # print ("outputs.shape", outputs.shape)

        targets = torch.from_numpy(y.astype(np.int64)).transpose(0, 1).contiguous().to(device)#.cuda()
        tt = torch.squeeze(targets.view(-1, model.batch_size))

        # LOSS COMPUTATION
        loss = [loss_fn(outputs.contiguous().view(-1, model.batch_size, model.vocab_size)[i], tt[i]).data.item() for i in range(model.seq_len)]

        losses.append(np.array(loss))

    return np.mean(losses, axis = 0)


###############################################################################
#
# RUN MAIN LOOP (TRAIN AND VAL)
#
###############################################################################

print("\n########## Getting validation average loss ##########################")


rnn = RNN(emb_size=200, hidden_size=1500,
            seq_len=35, batch_size=20,
            vocab_size=vocab_size, num_layers=2,
            dp_keep_prob=0.35)

gru = GRU(emb_size=200, hidden_size=1500,
            seq_len=35, batch_size=20,
            vocab_size=vocab_size, num_layers=2,
            dp_keep_prob=0.35)

trans = TRANSFORMER(vocab_size=vocab_size, n_units=512,
                    n_blocks=6, dropout=1.-0.9)
#
# # RNN
# num_epochs = args.num_epochs
# rnn.load_state_dict(torch.load('Question_4.1/RNN/best_params.pt'))
# rnn_name = rnn.__class__.__name__
# rnn_avg_losses = run_epoch(rnn, valid_data, model_type='RNN')
# rnn_filename = rnn_name +'_avg_losses.npy'
# path = os.path.join('Question_5.1', rnn_filename)
# np.save(path, rnn_avg_losses)

# GRU
gru.load_state_dict(torch.load('Question_4.1/GRU/best_params.pt'))
gru_name = gru.__class__.__name__
gru_avg_losses = run_epoch(gru, valid_data, model_type='GRU')
gru_filename = gru_name +'_avg_losses.npy'
path = os.path.join('Question_5.1', gru_filename)
np.save(path, gru_avg_losses)

# # Transformer
# trans.load_state_dict(torch.load('Question_4.1/TRANSFORMER/best_params.pt'))
# trans_name = trans.__class__.__name__
# trans_avg_losses = run_epoch(trans, valid_data, model_type='TRANSFORMER')
# trans_filename = trans_name +'_avg_losses.npy'
# path = os.path.join('Question_5.1', trans_filename)
# np.save(path, trans_avg_losses)

# plt.plot(rnn_avg_losses, label=rnn_name)
plt.plot(gru_avg_losses, label=gru_name)
# plt.plot(trans_avg_losses, label=trans_name)
plt.ylabel("average loss")
plt.xlabel("time steps")
plt.title("GRU Average loss per time-step")
plt.savefig('gru_avg_losses.png')
plt.clf()
print("\nDONE")
