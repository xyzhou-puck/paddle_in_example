#copyright (c) 2019 PaddlePaddle Authors. All Rights Reserve.
#
#Licensed under the Apache License, Version 2.0 (the "License");
#you may not use this file except in compliance with the License.
#You may obtain a copy of the License at
#
#    http://www.apache.org/licenses/LICENSE-2.0
#
#Unless required by applicable law or agreed to in writing, software
#distributed under the License is distributed on an "AS IS" BASIS,
#WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
#See the License for the specific language governing permissions and
#limitations under the License.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import paddle
import paddle.fluid as fluid

__all__ = ["VGGNet", "VGG19"]

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


class VGGNet():
    def __init__(self, layers=19):
        self.params = train_parameters
        self.layers = layers

    def net(self, input, class_dim=1000):
        layers = self.layers
        vgg_spec = {
            19: ([2, 2, 4, 4, 4])
        }
        assert layers in vgg_spec.keys(), \
            "supported layers are {} but input layer is {}".format(vgg_spec.keys(), layers)

        nums = vgg_spec[layers]
        block1, block1_conv1, _ = self.conv_block(input, 64, nums[0], name="conv1_")
        block2, block2_conv1, _ = self.conv_block(block1, 128, nums[1], name="conv2_")
        block3, block3_conv1, _ = self.conv_block(block2, 256, nums[2], name="conv3_")
        block4, block4_conv1, _ = self.conv_block(block3, 512, nums[3], name="conv4_")
        block5, block5_conv1, block5_conv2 = self.conv_block(block4, 512, nums[4], name="conv5_")

        outputs_dict = {
            'block1_conv1' : block1_conv1,
            'block2_conv1' : block2_conv1,
            'block3_conv1' : block3_conv1,
            'block4_conv1' : block4_conv1,
            'block5_conv1' : block5_conv1,
            'block5_conv2' : block5_conv2,
        }

        return outputs_dict

    def conv_block(self, input, num_filter, groups, name=None):
        conv = input
        conv1, conv2 = None, None
        for i in range(groups):
            conv = fluid.layers.conv2d(
                input=conv,
                num_filters=num_filter,
                filter_size=3,
                stride=1,
                padding=1,
                act='relu',
                param_attr=fluid.param_attr.ParamAttr(
                    name=name + str(i + 1) + "_weights"),
                bias_attr=False)
            if i == 0:
                conv1 = conv
            if i == 1:
                conv2 = conv
        pool = fluid.layers.pool2d(
            input=conv, pool_size=2, pool_type='max', pool_stride=2)
        return pool, conv1, conv2

def VGG19():
    model = VGGNet(layers=19)
    return model
