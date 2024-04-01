#!/usr/bin/env python
# -*- coding: utf-8 -*-
# @Time    : 2024/3/26 11:16
# @Author  : james.chen
# @File    : Code_of_Transformer.py
"""
https://blog.csdn.net/qq_37418807/article/details/120302612
此练习源于面试某大模型公司时，面试官让我手撕一个transformer，当时愣住了，所以回头打算写一遍。。。
切忌“知其然而不知其所以然”
"""

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as Func
import math, copy, time
from torch.autograd import Variable
import matplotlib.pyplot as plt
import seaborn
from IPython.display import Image


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


"""
Embedding module
contains input embedding and output embedding
"""


# Input embedding
# input: [batch_size, len]
# output: [batch_size, len, embedding_dim]
class Embeddings(nn.Module):
    def __init__(self, vocab, d_model):
        # d_model-- dim of embedding, vocab size of dictionary
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)


# Position Encoding
# input: [batch_size, len, embedding_dim]
# output: [batch_size, len, embedding_dim]
class PositionalEncoding(nn.Module):
    def __init__(self, d_model, drop_out, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.drop_out = nn.Dropout(p=drop_out)
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0., max_len).unsqueeze(1)
        div_term1 = torch.exp(torch.arange(0., d_model, 2)) * -(math.log(10000.0) / d_model)
        div_term2 = torch.exp(torch.arange(1., d_model, 2)) * -(math.log(10000.0) / d_model)
        pe[:, 0::2] = torch.sin(position * div_term1)
        pe[:, 1::2] = torch.cos(position * div_term2)
        pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + Variable(self.pe[:x.size(1), :], requires_grad=True)
        return self.drop_out(x)


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    # padding mask
    if mask:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = Func.softmax(scores, dim=-1)
    if dropout:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


class MultiHeadAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadAttention, self).__init__()
        assert d_model%h == 0
        self.d_k = d_model//h
        self.h = h
        self.linears = clones(nn.Linear(d_model,d_model),4)
        self.attn = None
        self.dropout=nn.Dropout(p=dropout)
    def forward(self,query,key,value,mask=None):
        if mask:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1,2) for l, x in zip(self.linears, (query,key,value))]



if __name__ == '__main__':
    # emb = Embeddings(1000, 10)
    # t = torch.Tensor([[1, 2, 3], [4, 5, 6]]).long()
    # e = emb(t)
    # print(e.shape)

    # plt.figure(figsize=(15, 5))
    # pe = PositionalEncoding(19, 0)
    # y = pe.forward(Variable(torch.zeros(1, 100, 19)))
    # print(y.shape)
    # plt.plot(np.arange(100), y[0, :, 4:8].data.numpy())
    # plt.legend(['dim %d' % p for p in [4, 5, 6, 7]])

    q = k = v = torch.ones([4, 8, 10, 100])
    attn = attention(q, k, v)
    print(attn[0].shape)
