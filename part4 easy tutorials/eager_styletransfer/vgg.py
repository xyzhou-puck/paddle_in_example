from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC, BatchNorm
from paddle.fluid.dygraph.base import to_variable

use_cudnn = False

train_parameters = {
    "input_size": [3, 224, 224],
    "input_mean": [0.485, 0.456, 0.406],
    "input_std": [0.229, 0.224, 0.225],
    "learning_strategy": {
        "name": "piecewise_decay",
        "batch_size": 256,
        "epochs": [30, 60, 90],
        "steps": [0.1, 0.01, 0.001, 0.0001]
    }
}

class conv_block_4(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_filter):
        super(conv_block_4, self).__init__(name_scope)

        self.conv1 = Conv2D(
            self.full_name(), 
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=fluid.param_attr.ParamAttr(
                name= "1_weights"),
            bias_attr=False)

        self.conv2 = Conv2D(
            self.full_name(), 
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=fluid.param_attr.ParamAttr(
                name= "2_weights"),
            bias_attr=False)

        self.conv3 = Conv2D(
            self.full_name(), 
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=fluid.param_attr.ParamAttr(
                name= "3_weights"),
            bias_attr=False)

        self.conv4 = Conv2D(
            self.full_name(), 
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=fluid.param_attr.ParamAttr(
                name= "4_weights"),
            bias_attr=False)

        self.pool1 = Pool2D(
            self.full_name(),
            pool_size=2, 
            pool_type='max', 
            pool_stride=2,
            use_cudnn=use_cudnn)

        self.num_filter = num_filter

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        conv3 = self.conv3(conv2)
        conv4 = self.conv4(conv3)
        x = self.pool1(conv4)
        return x, conv1, conv2

class conv_block_2(fluid.dygraph.Layer):
    def __init__(self, name_scope, num_filter):
        super(conv_block_2, self).__init__(name_scope)

        self.conv1 = Conv2D(
            self.full_name(), 
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=fluid.param_attr.ParamAttr(
                name= "1_weights"),
            bias_attr=False)

        self.conv2 = Conv2D(
            self.full_name(), 
            num_filters=num_filter,
            filter_size=3,
            stride=1,
            padding=1,
            act='relu',
            use_cudnn=use_cudnn,
            param_attr=fluid.param_attr.ParamAttr(
                name= "2_weights"),
            bias_attr=False)

        self.pool1 = Pool2D(
            self.full_name(),
            pool_size=2, 
            pool_type='max', 
            pool_stride=2,
            use_cudnn=use_cudnn)

        self.num_filter = num_filter

    def forward(self, inputs):
        conv1 = self.conv1(inputs)
        conv2 = self.conv2(conv1)
        x = self.pool1(conv2)
        return x, conv1, conv2


class build_vgg19(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(build_vgg19, self).__init__(name_scope)

        self.b1 = conv_block_2("block1", 64)
        self.b2 = conv_block_2("block2", 128)
        self.b3 = conv_block_4("block3", 256)
        self.b4 = conv_block_4("block4", 512)
        self.b5 = conv_block_4("block5", 512)

    def forward(self, inputs):
        x1, x1_c1, _ = self.b1(inputs)
        x2, x2_c1, _ = self.b2(x1)
        x3, x3_c1, _ = self.b3(x2)
        x4, x4_c1, _ = self.b4(x3)
        x5, x5_c1, x5_c2 = self.b5(x4)

        outputs_dict = {
            'block1_conv1' : x1_c1,
            'block2_conv1' : x2_c1,
            'block3_conv1' : x3_c1,
            'block4_conv1' : x4_c1,
            'block5_conv1' : x5_c1,
            'block5_conv2' : x5_c2,
        }

        return outputs_dict
