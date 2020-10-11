#!/usr/bin/env python
# -*- coding: utf-8 -*-
from __future__ import absolute_import
from __future__ import print_function

import torch.nn as nn
import torch
from torch.nn.parameter import Parameter
import math


# GCN implementation comes from https://github.com/wei-mao-2019/LearnTrajDep
class GraphConvolution(nn.Module):
    """
    adapted from : https://github.com/tkipf/gcn/blob/92600c39797c2bfb61a508e52b88fb554df30177/gcn/layers.py#L132
    """

    def __init__(self, in_features, out_features, bias=True, node_n=48):
        super(GraphConvolution, self).__init__()
        self.in_features = in_features
        self.out_features = out_features
        self.weight = Parameter(torch.FloatTensor(in_features, out_features))
        self.att = Parameter(torch.FloatTensor(node_n, node_n))
        if bias:
            self.bias = Parameter(torch.FloatTensor(out_features))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        self.att.data.uniform_(-stdv, stdv)
        if self.bias is not None:
            self.bias.data.uniform_(-stdv, stdv)

    def forward(self, input):
        support = torch.matmul(input, self.weight)
        output = torch.matmul(self.att, support)
        if self.bias is not None:
            return output + self.bias
        else:
            return output

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GC_Block(nn.Module):
    def __init__(self, in_features, p_dropout, bias=True, node_n=48):
        """
        Define a residual block of GCN
        """
        super(GC_Block, self).__init__()
        self.in_features = in_features
        self.out_features = in_features

        self.gc1 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn1 = nn.BatchNorm1d(node_n * in_features)

        self.gc2 = GraphConvolution(in_features, in_features, node_n=node_n, bias=bias)
        self.bn2 = nn.BatchNorm1d(node_n * in_features)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        y = self.gc2(y)
        b, n, f = y.shape
        y = self.bn2(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        return y + x

    def __repr__(self):
        return self.__class__.__name__ + ' (' \
               + str(self.in_features) + ' -> ' \
               + str(self.out_features) + ')'


class GCN(nn.Module):
    def __init__(self, input_feature, hidden_feature, p_dropout, num_stage=1, node_n=48):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super(GCN, self).__init__()
        self.num_stage = num_stage

        self.gc1 = GraphConvolution(input_feature, hidden_feature, node_n=node_n)
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.gc7 = GraphConvolution(hidden_feature, input_feature, node_n=node_n)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()

    def forward(self, x):
        y = self.gc1(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)
        y = self.do(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.gc7(y)
        y = y + x

        return y


# --------------------------------------------- INCEPTION MODULE ---------------------------------------------
class IdentityAutoencoder(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder = lambda x: x

        self.decoder = lambda x: x

    def forward(self, x):
        return x, x


class Conv1Channel(nn.Module):
    def __init__(self, nb_filters=1, filter_size=1, stride=1, dilation=1):
        super().__init__()
        self.conv = nn.Conv1d(1, nb_filters, filter_size, stride=stride, padding=0, dilation=dilation, groups=1,
                              bias=True, padding_mode='zeros')

    def forward(self, x):
        shape = x.shape
        x = x.reshape(-1, shape[-1])
        x = x[:, None, :]
        x = self.conv(x)
        x = x.reshape(shape[0], shape[1], -1)
        return x


class TemporalInceptionModule(nn.Module):
    def __init__(self):
        super().__init__()
        self.observed_length = [5, 5, 10, 10, 10]
        self.convolutions = nn.ModuleList([])

        # 5
        self.convolutions.append(Conv1Channel(nb_filters=12, filter_size=2))
        self.convolutions.append(Conv1Channel(nb_filters=9, filter_size=3))

        # 10
        self.convolutions.append(Conv1Channel(nb_filters=9, filter_size=3))
        self.convolutions.append(Conv1Channel(nb_filters=7, filter_size=5))
        self.convolutions.append(Conv1Channel(nb_filters=6, filter_size=7))

        self.output_size = self.forward(torch.ones(1, 1, 100)).shape[2]
        assert (len(self.observed_length) == len(self.convolutions))

    def forward(self, inpt):
        # Add the 10 last seen frame to the output features
        out = inpt[:, :, -10:]

        for obs_len, conv in zip(self.observed_length, self.convolutions):
            x = inpt[:, :, -obs_len:]
            y = conv(x)
            out = torch.cat((out, y), 2)

        return out


class InceptionGCN(nn.Module):
    def __init__(self, hidden_feature, p_dropout, num_stage=1, node_n=48, opt=None):
        """
        :param input_feature: num of input feature
        :param hidden_feature: num of hidden feature
        :param p_dropout: drop out prob.
        :param num_stage: number of residual blocks
        :param node_n: number of nodes in graph
        """
        super().__init__()
        self.opt = opt
        self.temporal_inception_mod = TemporalInceptionModule()

        # Overwrite input parameter with correct size with depends on the TIM
        hidden_feature = self.temporal_inception_mod.output_size

        self.num_stage = num_stage
        self.bn1 = nn.BatchNorm1d(node_n * hidden_feature)

        self.gcbs = []
        for i in range(num_stage):
            self.gcbs.append(GC_Block(hidden_feature, p_dropout=p_dropout, node_n=node_n))

        self.gcbs = nn.ModuleList(self.gcbs)

        self.do = nn.Dropout(p_dropout)
        self.act_f = nn.Tanh()
        self.final = nn.Linear(hidden_feature, opt.output_n + opt.input_n)

    def forward(self, x):
        x = x[:, :, :self.opt.input_n]
        y = self.temporal_inception_mod(x)
        b, n, f = y.shape
        y = self.bn1(y.view(b, -1)).view(b, n, f)
        y = self.act_f(y)

        for i in range(self.num_stage):
            y = self.gcbs[i](y)

        y = self.final(y)

        y = y + x[:, :, -1, None]

        return y
