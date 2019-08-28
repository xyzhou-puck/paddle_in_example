from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import time
import os
import random

import math
import io
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import reader

import shutil, sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

from base_model import BaseModel
import logging
import pickle

temp_model_path = './temp'
batch_size = 128
num_classes = 5
epochs = 5

def train():
    raw_data, raw_data_test = reader.get_lt5_data()

    model = BaseModel(fine_tune=False)
    loss, acc, output = model.build_graph()

    main_program = fluid.default_main_program()
    test_program = main_program.clone(for_test=True)

    optimizer = fluid.optimizer.Adadelta(0.01)
    optimizer.minimize(loss)

    place = fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    def prepare_input(batch, epoch_id=0):
        x, y = batch
        res = {}

        res['img'] = np.array(x).astype("float32") / 255
        res['label'] = np.array(y).astype("int64")

        return res

    def train_test(test_batch):
        total_acc = []
        input_data_feed = prepare_input(test_batch)
        fetch_outs = exe.run(program=test_program,
                             feed=input_data_feed,
                             fetch_list=[acc.name],
                             use_program_cache=True)

        acc_train = np.array(fetch_outs[0])
        total_acc.append(acc_train)
        print("test avg acc: {0:.2%}".format(np.mean(total_acc)))

    for epoch_id in range(epochs):
        print("epoch id", epoch_id)

        train_data_iter = reader.get_data_iter(raw_data, batch_size)
        test_data_iter  = reader.get_data_iter(raw_data_test, batch_size)

        data_iter = zip(train_data_iter, test_data_iter)

        total_loss = 0
        total_acc = []
        for batch_id, batch in enumerate(data_iter):
            batch_train, batch_test = batch
            input_data_feed = prepare_input(batch_train)
            fetch_outs = exe.run(program=main_program,
                                 feed=input_data_feed,
                                 fetch_list=[loss.name, acc.name],
                                 use_program_cache=True)

            cost_train = np.array(fetch_outs[0])
            acc_train = np.array(fetch_outs[1])
            total_loss += cost_train * batch_size
            total_acc.append(acc_train)

        print("train total loss: ", total_loss, np.mean(total_acc))
        train_test(batch_test)
        print()

    shutil.rmtree(temp_model_path, ignore_errors=True)
    os.makedirs(temp_model_path)
    fluid.io.save_params(executor=exe, dirname=temp_model_path)

def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == '__main__':
    print('training...')
    train()