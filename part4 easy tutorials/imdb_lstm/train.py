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

import numpy as np
import time
import os
import random

import math

import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from paddle.fluid.executor import Executor

import reader

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import os

from args import *
from base_model import BaseModel
from attention_model import AttentionModel
import logging
import pickle

SEED = 123

def train():
    args = parse_args()

    num_layers = args.num_layers
    src_vocab_size = args.src_vocab_size
    batch_size = args.batch_size
    dropout = args.dropout
    init_scale = args.init_scale
    max_grad_norm = args.max_grad_norm
    hidden_size = args.hidden_size

    model = BaseModel(
        hidden_size,
        src_vocab_size,
        batch_size,
        num_layers=num_layers,
        init_scale=init_scale,
        dropout=dropout)

    loss, acc = model.build_graph()
    # clone from default main program and use it as the validation program
    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone(for_test=True)

    lr = args.learning_rate
    opt_type = args.optimizer
    if opt_type == "sgd":
        optimizer = fluid.optimizer.SGD(lr)
    elif opt_type == "adam":
        optimizer = fluid.optimizer.Adam(lr)
    else:
        print("only support [sgd|adam]")
        raise Exception("opt type not support")

    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0) if args.use_gpu else fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    def prepare_input(batch, epoch_id=0, with_lr=True):
        src_ids, src_mask, label = batch
        res = {}
        src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1], 1))

        res['src'] = src_ids
        res['label'] = label
        res['src_sequence_length'] = src_mask

        return res

    all_data = reader.raw_data()
    
    max_epoch = args.max_epoch
    for epoch_id in range(max_epoch):
        start_time = time.time()
        print("epoch id", epoch_id)
        
        train_data_iter = reader.get_data_iter(all_data, batch_size)    

        total_loss = 0
        word_count = 0.0
        batch_id = 0
        for batch in train_data_iter:

            input_data_feed = prepare_input(batch)
            fetch_outs = exe.run(feed=input_data_feed,
                                 fetch_list=[loss.name, acc.name],
                                 use_program_cache=False)

            cost_train = np.array(fetch_outs[0])
            acc_train = np.array(fetch_outs[1])
            total_loss += cost_train

            if batch_id > 0 and batch_id % 100 == 0:
                print("current loss: %.3f, for step %d"  % (total_loss, batch_id))
                total_loss = 0.0

            batch_id += 1
    
    test_data_iter = reader.get_data_iter(all_data, batch_size, mode = 'test')

    all_acc = []

    for batch in test_data_iter:
        input_data_feed = prepare_input(batch)
        fetch_outs = exe.run(
            program = inference_program, 
            feed=input_data_feed,
            fetch_list=[acc.name],
            use_program_cache=False)

        all_acc.append(fetch_outs[0])

    all_acc = np.array(all_acc).astype("float32")

    print("test acc:%.3f" % all_acc.mean())

def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == '__main__':
    train()
