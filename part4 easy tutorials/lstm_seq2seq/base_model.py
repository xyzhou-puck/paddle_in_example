#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserve.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

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
                 hidden_size,
                 src_vocab_size,
                 tar_vocab_size,
                 batch_size,
                 num_layers=1,
                 init_scale=0.1,
                 dropout=None,
                 batch_first=True):

        self.hidden_size = hidden_size
        self.src_vocab_size = src_vocab_size
        self.tar_vocab_size = tar_vocab_size
        self.batch_size = batch_size
        self.num_layers = num_layers
        self.init_scale = init_scale
        self.dropout = dropout
        self.batch_first = batch_first

    def _build_data(self):
        self.src = layers.data(name="src", shape=[-1, 1, 1], dtype='int64')
        self.src_sequence_length = layers.data(
            name="src_sequence_length", shape=[-1], dtype='int32')

        self.tar = layers.data(name="tar", shape=[-1, 1, 1], dtype='int64')
        self.tar_sequence_length = layers.data(
            name="tar_sequence_length", shape=[-1], dtype='int32')
        self.label = layers.data(name="label", shape=[-1, 1, 1], dtype='int64')

    def _emebdding(self):
        self.src_emb = layers.embedding(
            input=self.src,
            size=[self.src_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='source_embedding',
                initializer=fluid.initializer.UniformInitializer(
                    low=-self.init_scale, high=self.init_scale)))
        self.tar_emb = layers.embedding(
            input=self.tar,
            size=[self.tar_vocab_size, self.hidden_size],
            dtype='float32',
            is_sparse=False,
            param_attr=fluid.ParamAttr(
                name='target_embedding',
                initializer=fluid.initializer.UniformInitializer(
                    low=-self.init_scale, high=self.init_scale)))

    def _build_encoder(self):
        self.enc_output, enc_last_hidden, enc_last_cell = basic_lstm( self.src_emb, None, None, self.hidden_size, num_layers=self.num_layers, batch_first=self.batch_first, \
                dropout_prob=self.dropout, \
                param_attr = ParamAttr( initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale) ), \
                bias_attr = ParamAttr( initializer = fluid.initializer.Constant(0.0) ), \
                sequence_length=self.src_sequence_length)

        return self.enc_output, enc_last_hidden, enc_last_cell

    def _build_decoder(self,
                       enc_last_hidden,
                       enc_last_cell,
                       mode='train'):
        softmax_weight = layers.create_parameter([self.hidden_size, self.tar_vocab_size], dtype="float32", name="softmax_weight", \
                    default_initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale))
        if mode == 'train':
            
            #fluid.layers.Print(self.tar_emb)
            #fluid.layers.Print(enc_last_hidden)
            #fluid.layers.Print(enc_last_cell)
            dec_output, dec_last_hidden, dec_last_cell = basic_lstm( self.tar_emb, enc_last_hidden, enc_last_cell, \
                    self.hidden_size, num_layers=self.num_layers, \
                    batch_first=self.batch_first, \
                    dropout_prob=self.dropout, \
                    param_attr = ParamAttr( initializer=fluid.initializer.UniformInitializer(low=-self.init_scale, high=self.init_scale) ), \
                    bias_attr = ParamAttr( initializer = fluid.initializer.Constant(0.0) ))

            dec_output = layers.matmul(dec_output, softmax_weight)

            return dec_output
        else:
            print("mode not supprt", mode)

    def _compute_loss(self, dec_output):
        loss = layers.softmax_with_cross_entropy(
            logits=dec_output, label=self.label, soft_label=False)

        loss = layers.reshape(loss, shape=[self.batch_size, -1])

        max_tar_seq_len = layers.shape(self.tar)[1]
        tar_mask = layers.sequence_mask(
            self.tar_sequence_length, maxlen=max_tar_seq_len, dtype='float32')
        loss = loss * tar_mask
        loss = layers.reduce_mean(loss, dim=[0])
        loss = layers.reduce_sum(loss)

        loss.permissions = True

        return loss

    def build_graph(self, mode='train'):
        if mode == 'train' or mode == 'eval':
            self._build_data()
            self._emebdding()
            enc_output, enc_last_hidden, enc_last_cell = self._build_encoder()
            dec_output = self._build_decoder(enc_last_hidden, enc_last_cell)

            loss = self._compute_loss(dec_output)
            return loss
        else:
            print("not support mode ", mode)
            raise Exception("not support mode: " + mode)
