from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers
import paddle.fluid as fluid
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN
import numpy as np
from paddle.fluid import ParamAttr

import ConvLSTM

class BaseModel(object):
    def __init__(self, 
                 batch_size=10,
                 filters=40, 
                 filter_size=3,
                 maxlen=15,
                 w=40,
                 h=40):
        self.batch_size = batch_size
        self.filters = filters
        self.filter_size = filter_size
        self.maxlen = maxlen
        self.w = w
        self.h = h

    def _build_data(self):
        self.input = fluid.layers.data(
            name='input', 
            shape=[-1, self.maxlen, self.h, self.w, 1], 
            dtype='float32')
        self.input_seqlen = fluid.layers.data(
            name='input_seqlen',
            shape=[-1],
            dtype='int64')
        self.label = fluid.layers.data(
            name='label', 
            shape=[-1, self.maxlen, self.h, self.w, 1], 
            dtype='float32')

    def _build_net(self):
        # ConvLSTM2D
        rnn_out, last_hidden = ConvLSTM.convlstm2d_rnn(
            rnn_input=self.input,
            init_hidden=None,
            init_cell=None,
            padding=1,
            hidden_h=self.h,
            hidden_w=self.w,
            filters=self.filters,
            filter_size=self.filter_size,
            sequence_length=self.input_seqlen)

        # Batch Norm
        bn = layers.layer_norm(rnn_out, begin_norm_axis=4)

        # ConvLSTM2D
        rnn_out, last_hidden = ConvLSTM.convlstm2d_rnn(
            rnn_input=bn,
            init_hidden=None,
            init_cell=None,
            padding=1,
            hidden_h=self.h,
            hidden_w=self.w,
            filters=self.filters,
            filter_size=self.filter_size,
            sequence_length=self.input_seqlen)

        # Batch Norm
        bn = layers.layer_norm(rnn_out, begin_norm_axis=4)

        # ConvLSTM2D
        rnn_out, last_hidden = ConvLSTM.convlstm2d_rnn(
            rnn_input=bn,
            init_hidden=None,
            init_cell=None,
            padding=1,
            hidden_h=self.h,
            hidden_w=self.w,
            filters=self.filters,
            filter_size=self.filter_size,
            sequence_length=self.input_seqlen)

        # Batch Norm
        bn = layers.layer_norm(rnn_out, begin_norm_axis=4)

        # ConvLSTM2D
        rnn_out, last_hidden = ConvLSTM.convlstm2d_rnn(
            rnn_input=bn,
            init_hidden=None,
            init_cell=None,
            padding=1,
            hidden_h=self.h,
            hidden_w=self.w,
            filters=self.filters,
            filter_size=self.filter_size,
            sequence_length=self.input_seqlen)

        # Batch Norm
        bn = layers.layer_norm(rnn_out, begin_norm_axis=4)
        
        # Transpose : (batch x C x D x H x W)
        tr = layers.transpose(bn, [0, 4, 1, 2, 3])

        # Conv3D
        conv3d = layers.conv3d(
            input=tr,
            num_filters=2,
            filter_size=3,
            padding=1)
        # conv3d : (batch x C x D x H x W)

        conv3d = layers.transpose(conv3d, [0, 2, 3, 4, 1])
        # conv3d: (batch x D x H x W x C)

        return conv3d

    def _compute_loss(self, pred):

        no_grad_set = []

        label = layers.cast(self.label, dtype="int64")
        label = layers.reshape(label, [-1, 1])
        pred = layers.reshape(pred, [-1, 2])

        no_grad_set.append(label.name)

        loss = layers.softmax_with_cross_entropy(pred, label)
        loss = layers.reshape(loss, shape=[self.batch_size, -1])
        loss = layers.reduce_mean(loss)

        return loss, no_grad_set

    def _compute_acc(self, pred):

        label = layers.cast(self.label, dtype="int64")
        label = layers.reshape(label, [-1, 1])
        pred = layers.reshape(pred, [-1, 2])

        acc = layers.accuracy(pred, label)

        return acc

    def build_graph(self, mode='train'):
        self._build_data()
        pred = self._build_net()
        if mode == 'train':
            loss, no_grad_set = self._compute_loss(pred)
            pred = layers.sigmoid(pred)
            acc = self._compute_acc(pred)
            return loss, acc, pred, no_grad_set
        else:
            pred = layers.sigmoid(pred)
            return pred



