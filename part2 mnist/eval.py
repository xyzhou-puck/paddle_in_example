#encoding=utf8

import os
import sys
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid

from arg_config import ArgConfig, print_arguments

def do_eval(args, preds):

    dataset = paddle.dataset.mnist.test()

    labels = []
    for feature, label in dataset():
        labels.append(label)

    labels = np.array(labels).astype("int32")

    acc = (preds == labels).mean()

    return acc

if __name__ == "__main__":
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    do_eval(args, [])

