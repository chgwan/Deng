# -*- coding: utf-8 -*-
import torch
import numpy as np
from torch import nn
import os
import pandas as pd

import sys
sys.path.append("/scratch/chgwang/Papers/magneticRecon/DRedModel")
import FedSeq # type: ignore package exits.

# set all nvidia card can use for training
os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"

def squash_packed(x, fn):
    return(torch.nn.utils.rnn.PackedSequence(fn(x.data), 
                x.batch_sizes, x.sorted_indices, x.unsorted_indices))

def builNetwork(layers):
    net = nn.ModuleList()
    for i in range(1, len(layers)):
        net.append(nn.GRU(layers[i-1], layers[i], 1, batch_first=True))
    return(net)

def sequence_mask(X, valid_len, value=0):
    """Mask irrelevant entries in sequences."""
    maxlen = X.size(1)
    # 新的 维度添加 [None, :] 在第0维添加新的维度
    mask = torch.arange((maxlen), dtype=torch.float32,
                        device=X.device)[None, :] < valid_len[:, None]
    X[~mask] = value
    return X
    
def masked_softmax(X, valid_len):
    """Perform softmax by filtering out some elements."""
    # X: 3-D tensor, valid_len: 1-D or 2-D tensor
    if valid_len is None:
        return nn.functional.softmax(X, dim=-1)
    else:
        shape = X.shape
        if valid_len.dim() == 1:
            valid_len = torch.repeat_interleave(valid_len, repeats=shape[1],
                                                dim=0)
        else:
            valid_len = valid_len.reshape(-1)
        # Fill masked elements with a large negative, whose exp is 0
        X = sequence_mask(X.reshape(-1, shape[-1]), valid_len, value=-1e6)
        return nn.functional.softmax(X.reshape(shape), dim=-1)

# Attention layer.
class MLPAttention(nn.Module):
    def __init__(self, query_size, key_size, units, dropout, **kwargs):
        super(MLPAttention, self).__init__(**kwargs)
        # no setting bias
        self.W_k = nn.Linear(key_size, units, bias=False)
        self.W_q = nn.Linear(query_size, units, bias=False)
        self.v = nn.Linear(units, 1, bias=False)
        self.dropout = nn.Dropout(dropout)

    def forward(self, query, key, value, valid_len):
        query, key = self.W_q(query), self.W_k(key)
        # Expand query to (`batch_size`, #queries, 1, units), and key to
        # (`batch_size`, 1, #kv_pairs, units). Then plus them with broadcast
        # query size (batch_size, 1, units)
        # key size (batch_size, length, units)
        # feature size (batch_size, 1, length, units)
        features = query.unsqueeze(2) + key.unsqueeze(1)
        features = torch.tanh(features)
        # reduce the last demension.
        # scores size (batch_size, 1, length)
        scores = self.v(features).squeeze(-1)
        attention_weights = self.dropout(masked_softmax(scores, valid_len))
        # value 
        return torch.bmm(attention_weights, value)

# Encoder Section
class Encoder(nn.Module):
    def __init__(self, *arg) -> None:
        super(Encoder, self).__init__()
        input_size = arg[0]
        output_size = arg[1]
        hiddens_size = arg[2]
        num_layers = arg[3]
        dropout = arg[4]
        self.encoder = nn.GRU(input_size, hiddens_size, num_layers, 
                        dropout=dropout, batch_first=True)
        self.dense = nn.Linear(hiddens_size, output_size)
    def forward(self, inputs, state=None):
        encoder_out, state = self.encoder(inputs)
        encoder_out = self.dense(encoder_out)
        return(encoder_out, state)
        
class Seq2SeqAttentionDecoder(nn.Module):
    def __init__(self, *arg):
        super(Seq2SeqAttentionDecoder, self).__init__()
        input_size = arg[0]
        output_size = arg[1]
        hiddens_size = arg[2]
        num_layers = arg[3]
        dropout = arg[4]
        self.decoder = nn.GRU(input_size, hiddens_size, num_layers, 
                        dropout=dropout, batch_first=True)
        self.dense = nn.Linear(hiddens_size, output_size)

    def init_state(self, enc_outputs, hidden_state, enc_valid_len):
        outputs, hidden_state = enc_outputs, hidden_state
        return (outputs, hidden_state, enc_valid_len)

    def forward(self, X, state):
        enc_outputs, hidden_state, enc_valid_len = state
        # X = self.embedding(X)
        # query shape: (batch_size, 1, num_hiddens)
        query = torch.unsqueeze(hidden_state[-1], dim=1)
        # context has same shape as query
        context = self.attention_cell(
            query, enc_outputs, enc_outputs, enc_valid_len)
        outputs = []
        for x in X:
            # Concatenate on the feature dimension
            out = torch.cat((context, torch.unsqueeze(x, dim=1)), dim=-1)
            out, state = self.decoder(out, state)
            outputs.append(out)
        outputs = torch.cat(outputs, dim=1)
        out, hidden_state = self.rnn(X, hidden_state)
        outputs = self.dense(out)
        return outputs [enc_outputs, hidden_state,
                                          enc_valid_len]
# Encoder Decoder with attention.                                  
class EncoderDecoder(nn.Module):
    """The base class for the encoder-decoder architecture."""
    def __init__(self, encoder, decoder, **kwargs):
        super(EncoderDecoder, self).__init__(**kwargs)
        self.encoder = encoder
        self.decoder = decoder

    def forward(self, enc_X, dec_X, *args):
        enc_outputs = self.encoder(enc_X, *args)
        dec_state = self.decoder.init_state(enc_outputs, *args)
        return self.decoder(dec_X, dec_state)
