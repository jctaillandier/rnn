import torch
import torch.nn as nn

import numpy as np
import torch.nn.functional as F
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
# Use the GPU if you have one
if torch.cuda.is_available():
    print("Using the GPU")
    device = torch.device("cuda")
else:
    print("WARNING: You are about to run on cpu, and this will likely run out \
      of memory. \n You can try setting batch_size=1 to reduce memory usage")
    device = torch.device("cpu")

# NOTE ==============================================
#
# Fill in code for every method which has a TODO
#
# Your implementation should use the contract (inputs
# and outputs) given for each model, because that is
# what the main script expects. If you modify the contract,
# you must justify that choice, note it in your report, and notify the TAs
# so that we run the correct code.
#
# You may modify the internals of the RNN and GRU classes
# as much as you like, except you must keep the methods
# in each (init_weights_uniform, init_hidden, and forward)
# Using nn.Module and "forward" tells torch which
# parameters are involved in the forward pass, so that it
# can correctly (automatically) set up the backward pass.
#
# You should not modify the interals of the Transformer
# except where indicated to implement the multi-head
# attention.


def clones(module, N):
    "A helper function for producing N identical layers (each with their own parameters)."
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

# Problem 1
class RNN(nn.Module): # Implement a stacked vanilla RNN with Tanh nonlinearities.
    def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
        super(RNN, self).__init__()

        #emb_size:     The numvwe of units in the input embeddings
        self.emb_size = emb_size

        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.seq_len = seq_len
        self.batch_size = batch_size
        #hidden_size:  The number of hidden units per layer
        self.hidden_size = hidden_size

        #seq_len:      The length of the input sequences
        #vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
        self.encoder = nn.Embedding(self.vocab_size, self.emb_size) # input is an integer, index of word in dict
        self.encoder = self.encoder.to(device)

        # To align sizes between embedding and first layer
        #self.first_layer = nn.Linear(self.emb_size, self.hidden_size)
        #self.first_layer = self.first_layer.to(device)

        self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
        self.decoder = self.decoder.to(device)
        #num_layers:   The depth of the stack (i.e. the number of hidden layers at
        #              each time-step)
        self.num_layers = num_layers

        #dp_keep_prob: The probability of *not* dropping out units in the
        #             non-recurrent connections.
        #            Do not apply dropout on recurrent connections.
        self.drop = nn.Dropout(1-dp_keep_prob)
        self.drop = self.drop.to(device)
        # Creating an array of layers of identical size
        # use module list inside clone()
        self.rec_layers = clones(nn.Linear(self.hidden_size, self.hidden_size), num_layers)
        self.rec_layers = self.rec_layers.to(device)

        self.regular_layers = clones(nn.Linear(self.hidden_size, self.hidden_size, bias=False), num_layers-1)
        #first layer to match embedding
        self.regular_layers.insert(0, nn.Linear(self.emb_size, self.hidden_size))
        self.regular_layers = self.regular_layers.to(device)

        self.hidden_states = [[],[]]
        #Initializing weights
        self.init_weights_uniform()



        # Initialization of the parameters of the recurrent and fc layers.
        # Your implementation should support any number of stacked hidden layers
        # (specified by num_layers), use an input embedding layer, and include fully
        # connected layers with dropout after each recurrent layer.
        # Note: you may use pytorch's nn.Linear, nn.Dropout, and nn.Embedding
        # modules, but not recurrent modules.
        # To create a variable number of parameter tensors and/or nn.Modules
        # (for the stacked hidden layer), you may need to use nn.ModuleList or the
        # provided clones function (as opposed to a regular python list), in order
        # for Pytorch to recognize these parameters as belonging to this nn.Module
        # and compute their gradients automatically. You're not obligated to use the
        # provided clones function.

    def init_weights_uniform(self):
        k = np.sqrt(1/self.hidden_size)
        # Initialize all the weights uniformed [-k, k]
        for index, data in enumerate(self.regular_layers):
            torch.nn.init.uniform_(data.weight.data, -k, k)

            torch.nn.init.uniform_(self.rec_layers[index].weight.data, -k, k)
            torch.nn.init.uniform_(self.rec_layers[index].bias.data, -k, k)

        # Embedding initialized uniform weights and zero bias
        torch.nn.init.uniform_(self.encoder.weight.data, -0.1, 0.1)
        # no bias in encoder
        #self.encoder.bias.data.fill_(0)

        # Output layer initialized uniform weights and zero bias
        torch.nn.init.uniform_(self.decoder.weight.data, a=-0.1, b=0.1)
        self.decoder.bias.data.fill_(0)


    def init_hidden(self):

        # initialize the hidden states to zero
        """
        This is used for the first mini-batch in an epoch, only.
        """
        #return  # a parameter tensor of shape (self.num_layers, self.batch_size, self.hidden_size)
        return torch.zeros((self.num_layers, self.batch_size, self.hidden_size))

    def forward(self, inputs, hidden):
        """
        Arguments:
            - inputs: A mini-batch of input sequences, composed of integers that
                        represent the index of the current token(s) in the vocabulary.
                            shape: (seq_len, batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
        Returns:
            - Logits for the softmax over output tokens at every time-step.
                **Do NOT apply softmax to the outputs!**
                Pytorch's CrossEntropyLoss function (applied in ptb-lm.py) does
                this computation implicitly.
                        shape: (seq_len, batch_size, vocab_size)
            - The final hidden states for every layer of the stacked RNN.
                These will be used as the initial hidden states for all the
                mini-batches in an epoch, except for the first, where the return
                value of self.init_hidden will be used.
                See the repackage_hiddens function in ptb-lm.py for more details,
                if you are curious.
                        shape: (num_layers, batch_size, hidden_size)
        """
        # Compute the forward pass, using a nested python for loops.
        # The outer for loop should iterate over timesteps, and the
        # inner for loop should iterate over hidden layers of the stack.
        #
        # Within these for loops, use the parameter tensors and/or nn.modules you
        # created in __init__ to compute the recurrent updates according to the
        # equations provided in the .tex of the assignment.
        #
        # Note that those equations are for a single hidden-layer RNN, not a stacked
        # RNN. For a stacked RNN, the hidden states of the l-th layer are used as
        # inputs to to the {l+1}-st layer (taking the place of the input sequence).

        # to store hidden states at each layers, time step (num_layers, batch_size, hidden_size)
        inputs = inputs.to(device)
        hidden_states = hidden
        hidden_states = hidden_states.to(device)

        logits = torch.empty(self.seq_len, self.batch_size, self.vocab_size)
        logits = logits.to(device)

        embedding = self.encoder(inputs)
        embedding = embedding.to(device)
        for l in range(self.num_layers):
            self.hidden_states[l].append(hidden[l,:,:])
        for timestep in range(inputs.shape[0]):  # Timesteps / word
            # Embedding returned a (seq_len, batch_size, emb_size)
            # I will iterate over each timestep where(axis==0)
            x = embedding[timestep,:]
            x = x.to(device)
            x = self.drop(x)

            for layer in range(len(self.regular_layers)): # hidden layers
                # pre activation:
                hid_temp = self.rec_layers[layer](self.hidden_states[layer][timestep])
                x = (self.regular_layers[layer](x) + hid_temp)

                # layer activation
                x = torch.tanh(x)
                # to use next timestep:
                hidden_states = x
                self.hidden_states[layer].append(x)
                x = self.drop(x)

            z = self.decoder(x)
            z = z.to(device)

            # z is shape (num_layers, batch_size, vocab size)
            #   We will want to apend the last layer (index=-1, : ,:) to logits at every timestep

            logits[timestep,:,:] = z[self.num_layers-1,:]

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden_states


    def generate(self, input, hidden, generated_seq_len):
        # Compute the forward pass, as in the self.forward method (above).
        # You'll probably want to copy substantial portions of that code here.
        #
        # We "seed" the generation by providing the first inputs.
        # Subsequent inputs are generated by sampling from the output distribution,
        # as described in the tex (Problem 5.3)
        # Unlike for self.forward, you WILL need to apply the softmax activation
        # function here in order to compute the parameters of the categorical
        # distributions to be sampled from at each time-step.

        """
        Arguments:
            - input: A mini-batch of input tokens (NOT sequences!)
                            shape: (batch_size)
            - hidden: The initial hidden states for every layer of the stacked RNN.
                            shape: (num_layers, batch_size, hidden_size)
            - generated_seq_len: The length of the sequence to generate.
                        Note that this can be different than the length used
                        for training (self.seq_len)
        """
        samples  = torch.empty(generated_seq_len, input.shape[0])
        samples = samples.to(device)
        samples[0,:] = input

        for timestep in range(generated_seq_len):
            x = self.encoder(input)
            x = self.drop(x)
            for layer in range(self.num_layers):
                    hid_temp = self.rec_layers[layer](hidden[layer,:,:])
                    x = (self.regular_layers[layer](x) + hid_temp)
                    x = torch.tanh(x)

                    hidden[layer,:,:] = x
                    self.drop(x)

            output = self.decoder(x)
            output = F.softmax(output, dim=0)
            output = torch.multinomial(output, 1)
            samples[timestep,:] = output.squeeze()

        return samples



# Problem 2
class GRU(nn.Module): # Implement a stacked GRU RNN
  """
  Follow the same instructions as for RNN (above), but use the equations for
  GRU, not Vanilla RNN.
  """
  def __init__(self, emb_size, hidden_size, seq_len, batch_size, vocab_size, num_layers, dp_keep_prob):
    super(GRU, self).__init__()

   #emb_size:     The numvwe of units in the input embeddings
    self.emb_size = emb_size

    self.vocab_size = vocab_size
    self.hidden_size = hidden_size
    self.seq_len = seq_len
    self.batch_size = batch_size
    #hidden_size:  The number of hidden units per layer
    self.hidden_size = hidden_size

    #seq_len:      The length of the input sequences
    #vocab_size:   The number of tokens in the vocabulary (10,000 for Penn TreeBank)
    self.encoder = nn.Embedding(self.vocab_size, self.emb_size) # input is an integer, index of word in dict
    self.encoder = self.encoder.to(device)
    #
    self.decoder = nn.Linear(self.hidden_size, self.vocab_size)
    self.decoder = self.decoder.to(device)

    #num_layers:   The depth of the stack (i.e. the number of hidden layers at
    #              each time-step)
    self.num_layers = num_layers

    #dp_keep_prob: The probability of *not* dropping out units in the
    #             non-recurrent connections.
    #            Do not apply dropout on recurrent connections.
    self.drop = nn.Dropout(1-dp_keep_prob)
    self.drop = self.drop.to(device)

    # Creating an array of layers of identical size
    self.u_reset_layers = clones(nn.Linear(self.hidden_size, self.hidden_size), self.num_layers)
    self.u_reset_layers = self.u_reset_layers.to(device)
    self.u_forget_layers = clones(nn.Linear(self.hidden_size, self.hidden_size), self.num_layers)
    self.u_forget_layers = self.u_forget_layers.to(device)
    self.u_hidden_layers = clones(nn.Linear(self.hidden_size, self.hidden_size), self.num_layers)
    self.u_hidden_layers = self.u_hidden_layers.to(device)

    self.reset_layers = clones(nn.Linear(self.hidden_size, self.hidden_size), self.num_layers-1).to(device)
    self.reset_layers.insert(0, nn.Linear(self.emb_size, self.hidden_size))
    self.reset_layers = self.reset_layers.to(device)

    self.forget_layers = clones(nn.Linear(self.hidden_size, self.hidden_size), self.num_layers-1).to(device)
    self.forget_layers.insert(0, nn.Linear(self.emb_size, self.hidden_size))
    self.forget_layers =  self.forget_layers.to(device)

    self.rec_layers = clones(nn.Linear(self.hidden_size, self.hidden_size), num_layers-1).to(device)
    self.rec_layers.insert(0, nn.Linear(self.emb_size, self.hidden_size))
    self.rec_layers = self.rec_layers.to(device)

    #Initializing weights
    self.init_weights_uniform()


  def init_weights_uniform(self):

        # Initialize all the weights uniformly in the range [-range, range]
        # and all the biases to 0 (in place)

        k = np.sqrt(1/self.hidden_size)

        for index, data in enumerate(self.rec_layers):
            torch.nn.init.uniform_(data.weight, -k, k)
            torch.nn.init.uniform_(data.bias, -k, k)

        for index, data in enumerate(self.reset_layers):
            torch.nn.init.uniform_(data.weight, -k, k)
            torch.nn.init.uniform_(data.bias, -k, k)

        for index, data in enumerate(self.forget_layers):
            torch.nn.init.uniform_(data.weight, -k, k)
            torch.nn.init.uniform_(data.bias, -k, k)

        for index, data in enumerate(self.u_reset_layers):
            torch.nn.init.uniform_(data.weight, -k, k)
            torch.nn.init.uniform_(data.bias, -k, k)

        for index, data in enumerate(self.u_forget_layers):
            torch.nn.init.uniform_(data.weight, -k, k)
            torch.nn.init.uniform_(data.bias, -k, k)

        for index, data in enumerate(self.u_hidden_layers):
            torch.nn.init.uniform_(data.weight, -k, k)
            torch.nn.init.uniform_(data.bias, -k, k)


        # Embedding initialized uniform weights and zero bias
        torch.nn.init.uniform_(self.encoder.weight.data, -0.1, 0.1)
        #self.encoder.bias.data.uniform_(-0.1, 0.1)

        # Output layer initialized uniform weights and zero bias
        torch.nn.init.uniform_(self.decoder.weight.data, a=-0.1, b=0.1)
        self.decoder.bias.data.fill_(0)

  def init_hidden(self):

    return torch.zeros((self.num_layers, self.batch_size, self.hidden_size))

  def forward(self, inputs, hidden):
        """
        seq_len: 35
        batch_size: 20
        lr: 20
        hidden_size: 200
        emb_size : 20
        """
        logits = torch.empty(self.seq_len, self.batch_size, self.vocab_size).to(device)
        inputs.to(device)
        embedding = self.encoder(inputs)
        embedding.to(device)
        self.hidden_states = [[],[]]

        for layer in range(self.num_layers):
            self.hidden_states[layer].append(hidden[layer,:,:])

        for timestep in range(inputs.shape[0]):
            x = self.drop(embedding[timestep,:])

            for layer in range(self.num_layers):
                reset = self.reset_layers[layer](x)
                u_reset_temp = self.u_reset_layers[layer](self.hidden_states[layer][timestep])
                reset_temp = torch.sigmoid(reset+u_reset_temp)

                forget = self.forget_layers[layer](x)
                u_forget_temp =self.u_forget_layers[layer](self.hidden_states[layer][timestep])
                forget_temp = torch.sigmoid(forget + u_forget_temp)

                h_wiggle = self.rec_layers[layer](x) + self.u_hidden_layers[layer](reset_temp * self.hidden_states[layer][timestep])
                h_wiggle = torch.tanh(h_wiggle)

                x = torch.mul((1-forget_temp),self.hidden_states[layer][timestep])+torch.mul(forget_temp,h_wiggle)
                self.hidden_states[layer].append(x)
                x = self.drop(x)


            logits[timestep,:,:] = self.decoder(x)

        #print('logits final size: ', logits.shape)

        return logits.view(self.seq_len, self.batch_size, self.vocab_size), hidden


  def generate(self, input, hidden, generated_seq_len):

    samples  = torch.empty(generated_seq_len, input.shape[0])
    samples = samples.to(device)
    samples[0,:] = input

    for timestep in range(generated_seq_len):
        x = self.encoder(input)
        x = self.drop(x)

        for layer in range(len(self.rec_layers)):
            reset = self.reset_layers[layer](x)
            u_reset_temp = self.u_reset_layers[layer](hidden[layer,:,:])
            reset_temp = torch.sigmoid(reset+u_reset_temp)

            forget = self.forget_layers[layer](x)
            u_forget_temp =self.u_forget_layers[layer](hidden[layer,:,:])
            forget_temp = torch.sigmoid(forget + u_forget_temp)

            h_wiggle = self.rec_layers[layer](x) + self.u_hidden_layers[layer](reset_temp * hidden[layer,:,:])
            h_wiggle = torch.tanh(h_wiggle)

            x = torch.mul((1-forget_temp),hidden[layer,:,:].clone())+torch.mul(forget_temp,h_wiggle)

            hidden[layer,:,:] = x
            self.drop(x)

        output = self.decoder(x)
        output = F.softmax(output, dim=0)
        output = torch.multinomial(output, 1)
        samples[timestep,:] = output.squeeze()

    return samples

# Problem 3
##############################################################################
#
# Code for the Transformer model
#
##############################################################################

"""
Implement the MultiHeadedAttention module of the transformer architecture.
All other necessary modules have already been implemented for you.
We're building a transfomer architecture for next-step prediction tasks, and
applying it to sequential language modelling. We use a binary "mask" to specify
which time-steps the model can use for the current prediction.
This ensures that the model only attends to previous time-steps.
The model first encodes inputs using the concatenation of a learned WordEmbedding
and a (in our case, hard-coded) PositionalEncoding.
The word embedding maps a word's one-hot encoding into a dense real vector.
The positional encoding 'tags' each element of an input sequence with a code that
identifies it's position (i.e. time-step).
These encodings of the inputs are then transformed repeatedly using multiple
copies of a TransformerBlock.
This block consists of an application of MultiHeadedAttention, followed by a
standard MLP; the MLP applies *the same* mapping at every position.
Both the attention and the MLP are applied with Resnet-style skip connections,
and layer normalization.
The complete model consists of the embeddings, the stacked transformer blocks,
and a linear layer followed by a softmax.
"""

#This code has been modified from an open-source project, by David Krueger.
#The original license is included below:
#MIT License
#
#Copyright (c) 2018 Alexander Rush
#
#Permission is hereby granted, free of charge, to any person obtaining a copy
#of this software and associated documentation files (the "Software"), to deal
#in the Software without restriction, including without limitation the rights
#to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
#copies of the Software, and to permit persons to whom the Software is
#furnished to do so, subject to the following conditions:
#
#The above copyright notice and this permission notice shall be included in all
#copies or substantial portions of the Software.
#
#THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
#IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
#FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
#AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
#LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
#OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
#SOFTWARE.



#----------------------------------------------------------------------------------
class MultiHeadedAttention(nn.Module):
    def __init__(self, n_heads, n_units, dropout=0.1):
        """
        n_heads: the number of attention heads
        n_units: the number of output units
        dropout: probability of DROPPING units

        """
        super(MultiHeadedAttention, self).__init__()
        # This sets the size of the keys, values, and queries (self.d_k) to all
        # be equal to the number of output units divided by the number of heads.
        self.n_heads = n_heads
        self.d_k = n_units // n_heads

        # This requires the number of n_heads to evenly divide n_units.
        assert n_units % n_heads == 0
        self.n_units = n_units
        self.drop = nn.Dropout(dropout)
        self.drop = self.drop


        self.w_k = nn.Linear(self.n_units, self.n_units)
        self.w_k = self.w_k
        self.w_q = nn.Linear(self.n_units, self.n_units)
        self.w_q = self.w_q
        self.w_v = nn.Linear(self.n_units, self.n_units)
        self.w_v = self.w_v

        self.w_o = nn.Linear(self.d_k*self.n_heads, self.n_units)
        self.w_o = self.w_o

        self.init_weights_uniform()

    def init_weights_uniform(self):
        # Initialize all the weights uniformly in the range [-range, range]
        # and all the biases to 0 (in place)
        k = np.sqrt(1/self.n_units)

        torch.nn.init.uniform_(self.w_k.weight, -k, k)
        torch.nn.init.uniform_(self.w_k.bias, -k, k)

        torch.nn.init.uniform_(self.w_v.weight, -k, k)
        torch.nn.init.uniform_(self.w_v.bias, -k, k)

        torch.nn.init.uniform_(self.w_q.weight, -k, k)
        torch.nn.init.uniform_(self.w_q.bias, -k, k)

        torch.nn.init.uniform_(self.w_o.weight, -k, k)
        torch.nn.init.uniform_(self.w_o.bias, -k, k)

    def forward(self, query, key, value, mask=None):
        # TODO: implement the masked multi-head attention.
        # query, key, and value all have size: (batch_size, seq_len, self.n_units)
        # mask has size: (batch_size, seq_len, seq_len)
        # As described in the .tex, apply input masking to the softmax
        # generating the "attention values" (i.e. A_i in the .tex)
        # Also apply dropout to the attention values.
        mask = mask.to(device, dtype=torch.float32)
        if mask is not None:
            mask = mask.unsqueeze(1)

        Q = self.w_q(query).view(query.size(0),query.size(1),self.n_heads,self.d_k).transpose(1,2)
        K = self.w_k(key).view(key.size(0),key.size(1),self.n_heads,self.d_k).transpose(1,2)
        z = torch.matmul(Q, K.transpose(-2, -1) )/ (np.sqrt(self.d_k))
        # z = z.to(device)
        # z is now the Attention value for this head

        # Mask and Softmax over inputs
        z = z.masked_fill(mask==0,-10e9)
        #z = (z*mask)-((10**9)*(1-mask))
        z =  F.softmax(z, dim=-1)

        # Full Head attention value
        z = torch.matmul(z, self.w_v(value).view(value.size(0),value.size(1),self.n_heads,self.d_k).transpose(1,2))
        #Then Dropout
        z = self.drop(z)

        # Output layer
        logits = self.w_o(z.transpose(1,2).contiguous().view(query.size(0),-1,self.n_units))

        return logits



#----------------------------------------------------------------------------------
# The encodings of elements of the input sequence

class WordEmbedding(nn.Module):
    def __init__(self, n_units, vocab):
        super(WordEmbedding, self).__init__()
        self.lut = nn.Embedding(vocab, n_units)
        self.n_units = n_units

    def forward(self, x):
        #print (x)
        return self.lut(x) * math.sqrt(self.n_units)


class PositionalEncoding(nn.Module):
    def __init__(self, n_units, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, n_units)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, n_units, 2).float() *
                             -(math.log(10000.0) / n_units))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:, :x.size(1)],
                         requires_grad=False)
        return self.dropout(x)



#----------------------------------------------------------------------------------
# The TransformerBlock and the full Transformer


class TransformerBlock(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerBlock, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(ResidualSkipConnectionWithLayerNorm(size, dropout), 2)

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask)) # apply the self-attention
        return self.sublayer[1](x, self.feed_forward) # apply the position-wise MLP


class TransformerStack(nn.Module):
    """
    This will be called on the TransformerBlock (above) to create a stack.
    """
    def __init__(self, layer, n_blocks): # layer will be TransformerBlock (below)
        super(TransformerStack, self).__init__()
        self.layers = clones(layer, n_blocks)
        self.layers = self.layers
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class FullTransformer(nn.Module):
    def __init__(self, transformer_stack, embedding, n_units, vocab_size):
        super(FullTransformer, self).__init__()
        self.transformer_stack = transformer_stack
        self.embedding = embedding
        self.output_layer = nn.Linear(n_units, vocab_size)

    def forward(self, input_sequence, mask):
        embeddings = self.embedding(input_sequence)
        return F.log_softmax(self.output_layer(self.transformer_stack(embeddings, mask)), dim=-1)


def make_model(vocab_size, n_blocks=6,
               n_units=512, n_heads=16, dropout=0.1):
    "Helper: Construct a model from hyperparameters."
    c = copy.deepcopy
    attn = MultiHeadedAttention(n_heads, n_units)
    ff = MLP(n_units, dropout)
    position = PositionalEncoding(n_units, dropout)
    model = FullTransformer(
        transformer_stack=TransformerStack(TransformerBlock(n_units, c(attn), c(ff), dropout), n_blocks),
        embedding=nn.Sequential(WordEmbedding(n_units, vocab_size), c(position)),
        n_units=n_units,
        vocab_size=vocab_size
        )

    # Initialize parameters with Glorot / fan_avg.
    for p in model.parameters():
        if p.dim() > 1:
            nn.init.xavier_uniform_(p)
    return model


#----------------------------------------------------------------------------------
# Data processing

def subsequent_mask(size):
    """ helper function for creating the masks. """
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Batch:
    "Object for holding a batch of data with mask during training."
    def __init__(self, x, pad=0):
        self.data = x
        self.mask = self.make_mask(self.data, pad)

    @staticmethod
    def make_mask(data, pad):
        "Create a mask to hide future words."
        mask = (data != pad).unsqueeze(-2)
        mask = mask & Variable(
            subsequent_mask(data.size(-1)).type_as(mask.data))
        return mask


#----------------------------------------------------------------------------------
# Some standard modules

class LayerNorm(nn.Module):
    "layer normalization, as in: https://arxiv.org/abs/1607.06450"
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class ResidualSkipConnectionWithLayerNorm(nn.Module):
    """
    A residual connection followed by a layer norm.
    Note for code simplicity the norm is first as opposed to last.
    """
    def __init__(self, size, dropout):
        super(ResidualSkipConnectionWithLayerNorm, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        "Apply residual connection to any sublayer with the same size."
        return x + self.dropout(sublayer(self.norm(x)))


class MLP(nn.Module):
    """
    This is just an MLP with 1 hidden layer
    """
    def __init__(self, n_units, dropout=0.1):
        super(MLP, self).__init__()
        self.w_1 = nn.Linear(n_units, 2048)
        self.w_2 = nn.Linear(2048, n_units)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))
