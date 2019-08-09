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
"""Utilities for parsing PTB text files."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import paddle
import random

Py3 = sys.version_info[0] == 3
UNK_ID = 0

def filter_len(src, max_sequence_len=80):

    if len(src) > max_sequence_len:
        src = src[:max_sequence_len]

    return src

def raw_data(max_sequence_len=80):

    word_dict = paddle.dataset.imdb.word_dict()
    train_set = paddle.dataset.imdb.train(word_dict)
    test_set = paddle.dataset.imdb.test(word_dict)

    train_src = []
    train_label = []

    test_src = []
    test_label = []

    temp = []

    for raw_data in train_set():
        temp.append((filter_len(raw_data[0]), [raw_data[1]]))

    random.shuffle(temp)

    for sent, label in temp:
        train_src.append(sent)
        train_label.append(label)

    for raw_data in test_set():
        test_src.append(filter_len(raw_data[0]))
        test_label.append([raw_data[1]])

    return train_src, train_label, test_src, test_label


def get_data_iter(raw_data, batch_size, mode='train', enable_ce=False):

    train_datas, train_labels, test_datas, test_labels = raw_data

    if mode == 'train':
        src_data = train_datas
        src_label = train_labels
    else:
        src_data = test_datas
        src_label = test_labels

    data_len = len(src_data)

    def to_pad_np(data, source=False):
        max_len = 0
        for ele in data:
            if len(ele) > max_len:
                max_len = len(ele)

        ids = np.ones((batch_size, max_len), dtype='int64') * 2
        mask = np.zeros((batch_size), dtype='int32')

        for i, ele in enumerate(data):
            ids[i, :len(ele)] = ele
            if not source:
                mask[i] = len(ele) - 1
            else:
                mask[i] = len(ele)

        return ids, mask

    b_src = []

    for j in range(data_len):
        if len(b_src) == batch_size:

            # sort
            #new_cache = sorted(b_src, key=lambda k: len(k[0]))

            batch_data = b_src

            sent_cache = [w[0] for w in batch_data]
            label = [w[1] for w in batch_data]
            sent, sent_mask = to_pad_np(sent_cache, source=True)
            label = np.array(label).astype("int64")

            yield (sent, sent_mask, label)

            b_src = []

        b_src.append((src_data[j], src_label[j]))
   
    """
    if len(b_src) > 0:

        batch_data = b_src

        sent_cache = [w[0] for w in batch_data]
        label = [w[1] for w in batch_data]
        sent, sent_mask = to_pad_np(sent_cache, source=True)
        label = np.array(label).astype("int64")

        yield (sent, sent_mask, label)
    """

