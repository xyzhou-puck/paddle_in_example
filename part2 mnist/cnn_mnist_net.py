#encoding=utf8

import os
import sys
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid

def create_net(
    is_training, 
    model_input, 
    args):
    """
    create the network
    """

    if is_training:
        img, label = model_input
    else:
        img = model_input

    conv_pool_1 = fluid.nets.simple_img_conv_pool(
        input=img,
        filter_size=args.conv1_filter_size,
        num_filters=args.conv1_filter_num,
        pool_size=args.pool1_size,
        pool_stride=args.pool1_stride,
        act=args.activity)

    conv_pool_2 = fluid.nets.simple_img_conv_pool(
        input=conv_pool_1,
        filter_size=args.conv2_filter_size,
        num_filters=args.conv2_filter_num,
        pool_size=args.pool2_size,
        pool_stride=args.pool2_stride,
        act=args.activity)

    prediction = fluid.layers.fc(input=conv_pool_2, size=args.class_num, act='softmax')

    if is_training:
        cost = fluid.layers.cross_entropy(input=prediction, label=label)
        avg_cost = fluid.layers.mean(cost)

        return avg_cost, prediction

    else:

        return prediction




