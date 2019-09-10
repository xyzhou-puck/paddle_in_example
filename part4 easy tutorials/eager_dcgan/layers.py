from __future__ import division
import paddle.fluid as fluid
import numpy as np
from paddle.fluid.dygraph.nn import Conv2D,  Conv2DTranspose , BatchNorm ,Pool2D
import os

# cudnn is not better when batch size is 1.
use_cudnn = True


class conv2d(fluid.dygraph.Layer):
    """docstring for Conv2D"""
    def __init__(self, 
                name_scope,
                num_filters=64,
                filter_size=7,
                stride=1,
                padding=0,
                norm=False,
                relu=False,
                relufactor=0.3,
                dropout=0.0,
                testing=False,
                use_bias=True):
        super(conv2d, self).__init__(name_scope)

        if use_bias == False:
            con_bias_attr = False
        else:
            con_bias_attr = fluid.initializer.Xavier()

        self.conv = Conv2D(
            self.full_name(),
            num_filters=num_filters,
            filter_size=filter_size,
            stride=stride,
            padding=padding,
            use_cudnn=use_cudnn,
            param_attr=fluid.initializer.Xavier(),
            bias_attr=con_bias_attr)

        if norm:
            self.bn = BatchNorm(
                self.full_name(),
                num_channels=num_filters,
                is_test=self.testing,
                param_attr=fluid.initializer.Xavier(),
                bias_attr=fluid.initializer.Xavier(),
                trainable_statistics=True)
    
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu
        self.dropout = dropout
        self.testing = testing

    
    def forward(self,inputs):
        conv = self.conv(inputs)
        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
        if self.dropout > 0.0:
            conv = fluid.layers.dropout(conv,dropout_prob=self.dropout, is_test=self.testing)
        return conv


class DeConv2D(fluid.dygraph.Layer):
    def __init__(self,
                 name_scope,
                 num_filters=64,
                 filter_size=5,
                 stride=1,
                 padding=[0,0],
                 outpadding=[0,0,0,0],
                 relu=False,
                 norm=False,
                 relufactor=0.3,
                 testing=False,
                 use_bias=True):
        super(DeConv2D,self).__init__(name_scope)

        if use_bias == False:
            de_bias_attr = False
        else:
            de_bias_attr = fluid.initializer.Xavier()

        self._deconv = Conv2DTranspose(self.full_name(),
                                       num_filters,
                                       filter_size=filter_size,
                                       stride=stride,
                                       padding=padding,
                                       param_attr=fluid.initializer.Xavier(),
                                       bias_attr=de_bias_attr)

        if norm:
            self.bn = BatchNorm(self.full_name(),
                                momentum=0.99,
                                is_test=testing,
                                num_channels=num_filters,
                                param_attr=fluid.initializer.Xavier(),
                                bias_attr=fluid.initializer.Xavier(),
                                trainable_statistics=True) 

        self.outpadding = outpadding
        self.relufactor = relufactor
        self.use_bias = use_bias
        self.norm = norm
        self.relu = relu
        self.testing = testing

    def forward(self,inputs):
        conv = fluid.layers.pad2d(inputs, 
            paddings=self.outpadding, 
            mode='constant', 
            pad_value=0.0)
        conv = self._deconv(conv)

        if self.norm:
            conv = self.bn(conv)
        if self.relu:
            conv = fluid.layers.leaky_relu(conv,alpha=self.relufactor)
        return conv