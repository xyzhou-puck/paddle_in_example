from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce
from keras.datasets import mnist

import re
import tarfile
import collections
import os
import io
import sys
import numpy as np

Py3 = sys.version_info[0] == 3

def get_lt5_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_lt5 = x_train[y_train < 5]
    y_train_lt5 = y_train[y_train < 5]
    x_test_lt5 = x_test[y_test < 5]
    y_test_lt5 = y_test[y_test < 5]

    return (x_train_lt5, y_train_lt5), (x_test_lt5, y_test_lt5)

def get_gte5_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()

    x_train_gte5 = x_train[y_train >= 5]
    y_train_gte5 = y_train[y_train >= 5] - 5
    x_test_gte5 = x_test[y_test >= 5]
    y_test_gte5 = y_test[y_test >= 5] - 5

    return (x_train_gte5, y_train_gte5), (x_test_gte5, y_test_gte5)

def get_data_iter(raw_data, batch_size):
    x, y = raw_data
    data_len = len(x)
    index = np.arange(data_len)

    b_src = []
    cache_num = 1

    for j in range(data_len):
        if len(b_src) == batch_size:

            new_cache = b_src

            for i in range(cache_num):
                batch_data = new_cache[i * batch_size:(i + 1) * batch_size]
                x_cache = [w[0] for w in batch_data]
                y_cache = [[w[1]] for w in batch_data]
                yield (x_cache, y_cache)

            b_src = []

        b_src.append((x[index[j]], y[index[j]]))
    if len(b_src) == batch_size * cache_num:

        new_cache = b_src

        for i in range(cache_num):
            batch_data = new_cache[i * batch_size:(i + 1) * batch_size]
            x_cache = [w[0] for w in batch_data]
            y_cache = [[w[1]] for w in batch_data]
            yield (x_cache, y_cache)