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

import sys
if sys.version[0] == '2':
    reload(sys)
    sys.setdefaultencoding("utf-8")
import os

import reader
from base_model import BaseModel
import logging
import pickle

def create_ngram_set(input_list, ngram_value=2):
    """
    Extract a set of n-grams from a list of integers.
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=2)
    {(4, 9), (4, 1), (1, 4), (9, 4)}
    >>> create_ngram_set([1, 4, 9, 4, 1, 4], ngram_value=3)
    [(1, 4, 9), (4, 9, 4), (9, 4, 1), (4, 1, 4)]
    """
    return set(zip(*[input_list[i:] for i in range(ngram_value)]))


def add_ngram(sequences, token_indice, ngram_range=2):
    """
    Augment the input list of list (sequences) by appending n-grams values.
    Example: adding bi-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017}
    >>> add_ngram(sequences, token_indice, ngram_range=2)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42]]
    Example: adding tri-gram
    >>> sequences = [[1, 3, 4, 5], [1, 3, 7, 9, 2]]
    >>> token_indice = {(1, 3): 1337, (9, 2): 42, (4, 5): 2017, (7, 9, 2): 2018}
    >>> add_ngram(sequences, token_indice, ngram_range=3)
    [[1, 3, 4, 5, 1337, 2017], [1, 3, 7, 9, 2, 1337, 42, 2018]]
    """
    new_sequences = []
    for input_list in sequences:
        new_list = input_list[:]
        for ngram_value in range(2, ngram_range + 1):
            for i in range(len(new_list) - ngram_value + 1):
                ngram = tuple(new_list[i:i + ngram_value])
                if ngram in token_indice:
                    new_list.append(token_indice[ngram])
        new_sequences.append(new_list)

    return new_sequences

def train():

    def prepare_input(batch):
        src_ids, label = batch
        res = {}

        res['src'] = src_ids
        res['label'] = label

        return res

    # Set parameters:
    # ngram_range = 2 will add bi-grams features
    ngram_range = 2
    max_features = 20000
    maxlen = 400
    batch_size = 32
    embedding_dims = 50
    epochs = 5

    print('Loading data...')
    all_data = reader.raw_data(num_words=max_features)
    x_train, y_train, x_test, y_test = all_data

    print(len(x_train), 'train sequences')
    print(len(x_test), 'test sequences')
    print('Average train sequence length: {}'.format(
        np.mean(list(map(len, x_train)), dtype=int)))
    print('Average test sequence length: {}'.format(
        np.mean(list(map(len, x_test)), dtype=int)))

    if ngram_range > 1:
        print('Adding {}-gram features'.format(ngram_range))
        # Create set of unique n-gram from the training set.
        ngram_set = set()
        for input_list in x_train:
            for i in range(2, ngram_range + 1):
                set_of_ngram = create_ngram_set(input_list, ngram_value=i)
                ngram_set.update(set_of_ngram)

        # Dictionary mapping n-gram token to a unique integer.
        # Integer values are greater than max_features in order
        # to avoid collision with existing features.
        start_index = max_features + 1
        token_indice = {v: k + start_index for k, v in enumerate(ngram_set)}
        indice_token = {token_indice[k]: k for k in token_indice}

        # max_features is the highest integer that could be found in the dataset.
        max_features = np.max(list(indice_token.keys())) + 1

        # Augmenting x_train and x_test with n-grams features
        x_train = add_ngram(x_train, token_indice, ngram_range)
        x_test = add_ngram(x_test, token_indice, ngram_range)
        print('Average train sequence length: {}'.format(
            np.mean(list(map(len, x_train)), dtype=int)))
        print('Average test sequence length: {}'.format(
            np.mean(list(map(len, x_test)), dtype=int)))

    print('Pad sequences (samples x time)')
    x_train = reader.pad_sequences(x_train, maxlen=maxlen)
    x_test = reader.pad_sequences(x_test, maxlen=maxlen)
    print('x_train shape:', x_train.shape)
    print('x_test shape:', x_test.shape)

    all_data = x_train, y_train, x_test, y_test

    print('Build model...')
    model = BaseModel(max_features=max_features)
    loss, acc = model.build_graph()

    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone(for_test=True)

    optimizer = fluid.optimizer.Adam(0.01)
    optimizer.minimize(loss)

    place = fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    for epoch_id in range(epochs):
        start_time = time.time()
        print("epoch id", epoch_id)
        
        train_data_iter = reader.get_data_iter(all_data, batch_size)    

        total_loss = 0
        total_acc = 0
        batch_id = 0
        for batch in train_data_iter:

            input_data_feed = prepare_input(batch)
            fetch_outs = exe.run(feed=input_data_feed,
                                 fetch_list=[loss.name, acc.name],
                                 use_program_cache=False)

            cost_train = np.array(fetch_outs[0])
            acc_train = np.array(fetch_outs[1])
            total_loss += cost_train
            total_acc += acc_train

            if batch_id > 0 and batch_id % 10 == 0:
                print("current loss: %.3f, current acc: %.3f for step %d"  % (total_loss, total_acc * 0.1, batch_id))
                total_loss = 0.0
                total_acc = 0.0

            batch_id += 1
    
    test_data_iter = reader.get_data_iter(all_data, batch_size, mode='test')

    all_acc = []

    for batch in test_data_iter:
        input_data_feed = prepare_input(batch)
        fetch_outs = exe.run(
            program = inference_program, 
            feed=input_data_feed,
            fetch_list=[loss.name, acc.name],
            use_program_cache=False)

        all_acc.append(fetch_outs[1])

    all_acc = np.array(all_acc).astype("float32")

    print("test acc: %.3f" % all_acc.mean())


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == '__main__':
    train()





