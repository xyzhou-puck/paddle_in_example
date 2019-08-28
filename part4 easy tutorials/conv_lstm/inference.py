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

batch_size = 1
n_frames = 15
n_samples = 1200
validation_split = 0.05
infer_model_path = './infer_model_new'

def train():

    model = BaseModel(batch_size=batch_size, maxlen=7)
    pred = model.build_graph(mode='test')

    inference_program = fluid.default_main_program().clone(for_test=True)

    place = fluid.CPUPlace()
    exe = Executor(place)
    exe.run(framework.default_startup_program())

    fluid.io.load_params(executor=exe, dirname=infer_model_path)

    def prepare_input(batch):
        x, y, x_seqlen = batch
        res = {}

        res['input'] = np.array(x).astype("float32")
        res['input_seqlen'] = np.array(x_seqlen).astype("int64")
        res['label'] = np.array(y).astype("float32")

        return res

    # (samples, seq, width, height, pixel)
    noisy_movies, shifted_movies = reader.generate_movies(n_samples, n_frames)

    # Testing the network on one movie
    # feed it with the first 7 positions and then
    # predict the new positions
    which = 1004
    track_test = noisy_movies[which][:7, ::, ::, ::]
    track_res = shifted_movies[which][:7, ::, ::, ::]

    track_test = track_test[np.newaxis, ::, ::, ::, ::]
    track_res = track_res[np.newaxis, ::, ::, ::, ::]

    for j in range(16):

        track_raw = track_test, track_res

        data_iter = reader.get_data_iter(track_raw, 1) 

        # batch
        for batch in data_iter:
            input_data_feed = prepare_input(batch)
            fetch_outs = exe.run(program=inference_program,
                                 feed=input_data_feed,
                                 fetch_list=[pred.name],
                                 use_program_cache=False)

            guess = fetch_outs[0]
            last_seq = guess[0][-1]

            temp = []
            for row in last_seq:
                temp_row = []
                for ele in row:
                    pred_label = np.argsort(ele)[1]
                    temp_row.append([pred_label])
                temp.append(temp_row)

            guess = [[temp]]
            new = np.array(guess)
            track_test = np.concatenate((track_test, new), axis=1)


    # And then compare the predictions
    # to the ground truth
    track2 = noisy_movies[which][::, ::, ::, ::]
    for i in range(15):
        fig = plt.figure(figsize=(10, 5))

        ax = fig.add_subplot(121)

        if i >= 7:
            ax.text(1, 3, 'Predictions !', fontsize=20, color='w')
        else:
            ax.text(1, 3, 'Initial trajectory', fontsize=20)

        toplot = track_test[0][i, ::, ::, 0]

        plt.imshow(toplot)
        ax = fig.add_subplot(122)
        plt.text(1, 3, 'Ground truth', fontsize=20)

        toplot = track2[i, ::, ::, 0]
        if i >= 2:
            toplot = shifted_movies[which][i - 1, ::, ::, 0]

        plt.imshow(toplot)
        plt.savefig('./res/%i_animate.png' % (i + 1))


def get_cards():
    num = 0
    cards = os.environ.get('CUDA_VISIBLE_DEVICES', '')
    if cards != '':
        num = len(cards.split(","))
    return num


if __name__ == '__main__':
    train()






















