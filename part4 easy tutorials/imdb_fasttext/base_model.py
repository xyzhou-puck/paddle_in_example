from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers
import paddle.fluid as fluid
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN
import numpy as np
from paddle.fluid import ParamAttr
from paddle.fluid.contrib.layers import basic_lstm, BasicLSTMUnit

INF = 1. * 1e5
alpha = 0.6

class BaseModel(object):
    def __init__(self,
                 max_features,
                 embedding_dims=50,
                 maxlen=400,
                 batch_size=32,
                 init_scale=0.1,
                 dropout=None,
                 batch_first=True):

        self.max_features = max_features
        self.embedding_dims = embedding_dims
        self.maxlen = maxlen
        self.batch_size = batch_size
        self.init_scale = init_scale
        self.dropout = dropout
        self.batch_first = batch_first

    def _build_data(self):
        self.src = layers.data(name="src", shape=[1], dtype='int64', lod_level=1)
        self.label = layers.data(name="label", shape=[-1, 1], dtype='int64')

    def _emebdding(self):

        self.src_emb = layers.embedding(
            input=self.src,
            size=[self.max_features, self.embedding_dims])

    def _build_net(self):

        self.pool1 = layers.sequence_pool(
            input=self.src_emb,
            pool_type="average")

        self.output = layers.fc(self.pool1, 2)
        self.output = layers.softmax(self.output)

        return self.output
    
    def _compute_loss_acc(self, pred):

        loss = layers.cross_entropy(
            pred, 
            label=self.label,
            soft_label=False)

        loss = layers.reshape(loss, shape=[self.batch_size, -1])
        loss = layers.reduce_mean(loss)
        acc = fluid.layers.accuracy(input=pred, label=self.label)

        return loss, acc

    def build_graph(self):
        self._build_data()
        self._emebdding()
        output = self._build_net()
        return self._compute_loss_acc(output)
