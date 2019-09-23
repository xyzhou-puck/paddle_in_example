from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import os
import sys
import numpy as np

def get_data_iter(raw_data, batch_size, mode='train', enable_ce=False):

    src_data, tar_data = raw_data

    data_len = len(src_data)

    index = np.arange(data_len)
#    if mode == "train" and not enable_ce:
#        np.random.shuffle(index)

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

    cache_num = 20
    if mode != "train":
        cache_num = 1
    for j in range(data_len):
        if len(b_src) == batch_size * cache_num:
            # build batch size

            # sort
            new_cache = sorted(b_src, key=lambda k: len(k[0]))

            for i in range(cache_num):
                batch_data = new_cache[i * batch_size:(i + 1) * batch_size]
                src_cache = [w[0] for w in batch_data]
                tar_cache = [w[1] for w in batch_data]
                src_ids, src_mask = to_pad_np(src_cache, source=True)
                tar_ids, tar_mask = to_pad_np(tar_cache)

                #print( "src ids", src_ids )
                yield (src_ids, src_mask, tar_ids, tar_mask)

            b_src = []

        b_src.append((src_data[index[j]], tar_data[index[j]]))
    if len(b_src) == batch_size * cache_num:
        new_cache = sorted(b_src, key=lambda k: len(k[0]))

        for i in range(cache_num):
            batch_data = new_cache[i * batch_size:(i + 1) * batch_size]
            src_cache = [w[0] for w in batch_data]
            tar_cache = [w[1] for w in batch_data]
            src_ids, src_mask = to_pad_np(src_cache, source=True)
            tar_ids, tar_mask = to_pad_np(tar_cache)

            #print( "src ids", src_ids )
            yield (src_ids, src_mask, tar_ids, tar_mask)

def raw_data(data_path, num_samples=100, max_sequence_len=100):
    # Vectorize the data.
    input_texts = []
    target_texts = []
    input_characters = set()
    target_characters = set()
    with open(data_path, 'r', encoding='utf-8') as f:
        lines = f.read().split('\n')
    for line in lines[: min(num_samples, len(lines) - 1)]:
        input_text, target_text = line.split('\t')
        # We use "tab" as the "start sequence" character
        # for the targets, and "\n" as "end sequence" character.
        target_text = '\t' + target_text + '\n'
        input_texts.append(input_text)
        target_texts.append(target_text)
        for char in input_text:
            if char not in input_characters:
                input_characters.add(char)
        for char in target_text:
            if char not in target_characters:
                target_characters.add(char)

    input_characters = sorted(list(input_characters))
    target_characters = sorted(list(target_characters))
    num_encoder_tokens = len(input_characters)
    num_decoder_tokens = len(target_characters)
    max_encoder_seq_length = max([len(txt) for txt in input_texts])
    max_decoder_seq_length = max([len(txt) for txt in target_texts])

    print('Number of samples:', len(input_texts))
    print('Number of unique input tokens:', num_encoder_tokens)
    print('Number of unique output tokens:', num_decoder_tokens)
    print('Max sequence length for inputs:', max_encoder_seq_length)
    print('Max sequence length for outputs:', max_decoder_seq_length)

    input_token_index = dict([(char, i) for i, char in enumerate(input_characters)])
    target_token_index = dict([(char, i) for i, char in enumerate(target_characters)])

    input_data = []
    target_data = []

    for i, input_text in enumerate(input_texts):
         input_text_data = []
         for j, char in enumerate(input_text):
             input_text_data.append(input_token_index[char])
         input_data.append(input_text_data)

    for i, target_text in enumerate(target_texts):
         target_text_data = []
         for j, char in enumerate(target_text):
             target_text_data.append(target_token_index[char])
         target_data.append(target_text_data)

    vars = dict()
    vars['num_encoder_tokens'] = num_encoder_tokens
    vars['num_decoder_tokens'] = num_decoder_tokens
    vars['max_encoder_seq_length'] = max_encoder_seq_length
    vars['max_decoder_seq_length'] = max_decoder_seq_length

    return [(input_data, target_data), vars]
