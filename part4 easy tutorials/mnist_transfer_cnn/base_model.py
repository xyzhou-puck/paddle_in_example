from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle.fluid.layers as layers
import paddle.fluid as fluid
from paddle.fluid.layers.control_flow import StaticRNN as PaddingRNN
import numpy as np
from paddle.fluid import ParamAttr

class BaseModel(object):
    def __init__(self, filters=32, pool_size=2, filter_size=3, classes=5, fine_tune=False):
        self.filters = filters
        self.pool_size = pool_size
        self.filter_size = filter_size
        self.classes = classes
        self.trainable = not fine_tune

    def _build_data(self):
        self.img = fluid.layers.data(name='img', shape=[-1, 28, 28], dtype='float32')
        self.label = fluid.layers.data(name='label', shape=[-1, 1], dtype='int64')

    def _feature_layers(self):
        self.reshape1 = fluid.layers.reshape(self.img, [-1, 1, 28, 28])
        self.conv1 = fluid.layers.conv2d(
            input=self.reshape1,
            num_filters=self.filters,
            filter_size=self.filter_size,
            act='relu',
            param_attr=fluid.ParamAttr(name='conv1', trainable=self.trainable))
        self.conv2 = fluid.layers.conv2d(
            input=self.conv1,
            num_filters=self.filters,
            filter_size=self.filter_size,
            act='relu',
            param_attr=fluid.ParamAttr(name='conv2', trainable=self.trainable))
        self.pool1 = fluid.layers.pool2d(
            input=self.conv2,
            pool_size=self.pool_size,
            pool_type='max')
        self.dropout1 = fluid.layers.dropout(self.pool1, 0.25)
        self.feature_out = fluid.layers.flatten(self.dropout1)
        return self.feature_out

    def _classification_layers(self):
        self.fc1 = fluid.layers.fc(
            input=self.feature_out, 
            size=128,
            act='relu')
        self.dropout2 = fluid.layers.dropout(self.fc1, 0.5)
        self.fc2 = fluid.layers.fc(
            input=self.dropout2, 
            size=self.classes,
            act='softmax')
        return self.fc2

    def _compute_loss(self, output):
        print(output.shape)
        print(self.label.shape)
        loss = layers.cross_entropy(output, label=self.label, soft_label=False)
        loss = layers.reduce_mean(loss)
        return loss

    def _compute_acc(self, output):
        acc = layers.accuracy(input=output, label=self.label)
        return acc

    def build_graph(self):
        self._build_data()
        self._feature_layers()
        if self.trainable:
            self.feature_out.stop_gradient = True
        out = self._classification_layers()
        loss = self._compute_loss(out)
        acc = self._compute_acc(out)
        return loss, acc, out







