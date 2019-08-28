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
from visualdl import LogWriter

import reader

import shutil, sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")

from base_model import BaseModel
import logging
import pickle
import pylab as plt

max_epoch = 20
batch_size = 10
n_frames = 15
n_samples = 1200
validation_split = 0.05

log_path = './vdl_log_new'
params_path = './infer_model_new'

def split(data, split):
    x, y = data
    size = len(x)
    valid_size = int(size * split)
    validation_data = x[:valid_size], y[:valid_size]
    train_data = x[valid_size:], y[valid_size:]

    return train_data, validation_data

def train():

    model = BaseModel(batch_size=batch_size, maxlen=n_frames)
    loss, acc, output, no_grad_set = model.build_graph()

    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adadelta(0.001)
    optimizer.minimize(loss, no_grad_set=no_grad_set)

    place = fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    log_writter = LogWriter(log_path, sync_cycle=10)  

    with log_writter.mode("train") as logger:          
        log_train_loss = logger.scalar(tag="train_loss") 
        log_train_acc = logger.scalar(tag="train_acc")

    with log_writter.mode("validation") as logger:
        log_valid_loss = logger.scalar(tag="validation_loss")
        log_valid_acc = logger.scalar(tag="validation_acc")

    def prepare_input(batch):
        x, y, x_seqlen = batch
        res = {}

        res['input'] = np.array(x).astype("float32")
        res['input_seqlen'] = np.array(x_seqlen).astype("int64")
        res['label'] = np.array(y).astype("float32")

        return res

    # (samples, seq, width, height, pixel)
    noisy_movies, shifted_movies = reader.generate_movies(n_samples, n_frames)
    data = noisy_movies[:1000], shifted_movies[:1000]
    train_data, validation_data = split(data, validation_split)

    step_id = 0
    for epoch_id in range(max_epoch):
        start_time = time.time()
        print("epoch id", epoch_id)

        valid_data_iter = reader.get_data_iter(validation_data, batch_size) 
        train_data_iter = reader.get_data_iter(train_data, batch_size) 

        # train
        total_loss = 0
        batch_id = 0
        for batch in train_data_iter:
            input_data_feed = prepare_input(batch)
            fetch_outs = exe.run(program=main_program,
                                 feed=input_data_feed,
                                 fetch_list=[loss.name, acc.name],
                                 use_program_cache=False)

            cost_train = np.array(fetch_outs[0])
            acc_train = fetch_outs[1]
            total_loss += cost_train

            if batch_id > 0 and batch_id % 5 == 0:
                log_train_loss.add_record(step_id, total_loss) 
                log_train_acc.add_record(step_id, acc_train)
                step_id += 1
                print("current loss: %.7f, for batch %d"  % (total_loss, batch_id))
                total_loss = 0.0

            batch_id += 1


        # validate
        total_loss = 0
        total_acc = 0
        batch_id = 0
        for batch in valid_data_iter:
            input_data_feed = prepare_input(batch)
            fetch_outs = exe.run(program=inference_program,
                                 feed=input_data_feed,
                                 fetch_list=[loss.name, acc.name],
                                 use_program_cache=False)

            cost_train = np.array(fetch_outs[0])
            acc_train = fetch_outs[1]
            total_loss += cost_train
            batch_id += 1

        log_valid_loss.add_record(epoch_id, total_loss)
        log_valid_acc.add_record(epoch_id, total_acc / batch_id)
        print("validation loss: %.7f"  % (total_loss))

    fluid.io.save_inference_model(
        dirname=params_path,
        feeded_var_names=['input', 'input_seqlen'], 
        target_vars=[loss, acc], 
        executor=exe)

def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == '__main__':
    train()











