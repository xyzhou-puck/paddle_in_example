from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np
import paddle
import paddle.fluid as fluid
import random
import json
from six.moves import range
import six
from keras.utils.data_utils import get_file

Py3 = sys.version_info[0] == 3
UNK_ID = 0

# FROM KERAS SOURCE CODE
# https://github.com/keras-team/
# keras-preprocessing/blob/master/keras_preprocessing/sequence.py
def remove_long_seq(maxlen, seq, label):
    new_seq, new_label = [], []
    for x, y in zip(seq, label):
        if len(x) < maxlen:
            new_seq.append(x)
            new_label.append(y)
    return new_seq, new_label

# FROM KERAS SOURCE CODE
# https://github.com/keras-team/
# keras-preprocessing/blob/master/keras_preprocessing/sequence.py

def pad_sequences(sequences, maxlen=None, dtype='int32',
                  padding='pre', truncating='pre', value=0.):

    if not hasattr(sequences, '__len__'):
        raise ValueError('`sequences` must be iterable.')
    num_samples = len(sequences)

    lengths = []
    for x in sequences:
        try:
            lengths.append(len(x))
        except TypeError:
            raise ValueError('`sequences` must be a list of iterables. '
                             'Found non-iterable: ' + str(x))

    if maxlen is None:
        maxlen = np.max(lengths)

    # take the sample shape from the first non empty sequence
    # checking for consistency in the main loop below.
    sample_shape = tuple()
    for s in sequences:
        if len(s) > 0:
            sample_shape = np.asarray(s).shape[1:]
            break

    is_dtype_str = np.issubdtype(dtype, np.str_) or np.issubdtype(dtype, np.unicode_)
    if isinstance(value, six.string_types) and dtype != object and not is_dtype_str:
        raise ValueError("`dtype` {} is not compatible with `value`'s type: {}\n"
                         "You should set `dtype=object` for variable length strings."
                         .format(dtype, type(value)))

    x = np.full((num_samples, maxlen) + sample_shape, value, dtype=dtype)
    for idx, s in enumerate(sequences):
        if not len(s):
            continue  # empty list/array was found
        if truncating == 'pre':
            trunc = s[-maxlen:]
        elif truncating == 'post':
            trunc = s[:maxlen]
        else:
            raise ValueError('Truncating type "%s" '
                             'not understood' % truncating)

        # check `trunc` has expected shape
        trunc = np.asarray(trunc, dtype=dtype)
        if trunc.shape[1:] != sample_shape:
            raise ValueError('Shape of sample %s of sequence at position %s '
                             'is different from expected shape %s' %
                             (trunc.shape[1:], idx, sample_shape))

        if padding == 'post':
            x[idx, :len(trunc)] = trunc
        elif padding == 'pre':
            x[idx, -len(trunc):] = trunc
        else:
            raise ValueError('Padding type "%s" not understood' % padding)
    return x

# FROM KERAS SOURCE CODE
# https://github.com/keras-team/keras/blob/master/keras/datasets/imdb.py
def raw_data(maxlen=None, num_words=20000):
    path='imdb.npz'
    skip_top=0
    seed=113
    start_char=1 
    oov_char=2
    index_from=3

    path = get_file(path,
                    origin='https://s3.amazonaws.com/text-datasets/imdb.npz',
                    file_hash='599dadb1135973df5b59232a0e9a887c')
    with np.load(path, allow_pickle=True) as f:
        x_train, labels_train = f['x_train'], f['y_train']
        x_test, labels_test = f['x_test'], f['y_test']

    rng = np.random.RandomState(seed)
    indices = np.arange(len(x_train))
    rng.shuffle(indices)
    x_train = x_train[indices]
    labels_train = labels_train[indices]

    indices = np.arange(len(x_test))
    rng.shuffle(indices)
    x_test = x_test[indices]
    labels_test = labels_test[indices]

    xs = np.concatenate([x_train, x_test])
    labels = np.concatenate([labels_train, labels_test])

    if start_char is not None:
        xs = [[start_char] + [w + index_from for w in x] for x in xs]
    elif index_from:
        xs = [[w + index_from for w in x] for x in xs]

    if maxlen:
        xs, labels = _remove_long_seq(maxlen, xs, labels)
        if not xs:
            raise ValueError('After filtering for sequences shorter than maxlen=' +
                             str(maxlen) + ', no sequence was kept. '
                             'Increase maxlen.')
    if not num_words:
        num_words = max([max(x) for x in xs])

    # by convention, use 2 as OOV word
    # reserve 'index_from' (=3 by default) characters:
    # 0 (padding), 1 (start), 2 (OOV)
    if oov_char is not None:
        xs = [[w if (skip_top <= w < num_words) else oov_char for w in x]
              for x in xs]
    else:
        xs = [[w for w in x if skip_top <= w < num_words]
              for x in xs]

    idx = len(x_train)
    x_train, y_train = np.array(xs[:idx]), np.array(labels[:idx])
    x_test, y_test = np.array(xs[idx:]), np.array(labels[idx:])

    return x_train, y_train, x_test, y_test
    

def get_data_iter(raw_data, batch_size, mode='train', enable_ce=False):

    train_datas, train_labels, test_datas, test_labels = raw_data

    if mode == 'train':
        src_data = train_datas
        src_label = train_labels
    else:
        src_data = test_datas
        src_label = test_labels

    data_len = len(src_data)
    b_src = []

    def to_lod_tensor(input_data):
        length = []
        data = []
        for i, sentence in enumerate(input_data):
            length.append(len(sentence))
            for word in sentence:
                data.append([word])

        lod_data = fluid.create_lod_tensor(
            np.array(data).astype('int64'),
            [length],
            fluid.CPUPlace())

        return lod_data

    for j in range(data_len):
        if len(b_src) == batch_size:

            batch_data = b_src

            sent_cache = [w[0] for w in batch_data]
            label = [[w[1]] for w in batch_data]

            sent = to_lod_tensor(sent_cache)
            label = np.array(label).astype("int64")

            # (lod_tensor, tensor)
            yield (sent, label)

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

