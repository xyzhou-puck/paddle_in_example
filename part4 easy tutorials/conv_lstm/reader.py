from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from functools import reduce

import re
import tarfile
import collections
import os
import io
import sys
import random
import numpy as np
import paddle.fluid as fluid

def get_data_iter(raw_data, batch_size):

    x, y = raw_data
    data_len = len(x)

    temp_data = []
    for i in range(data_len):
        temp_data.append((x[i], y[i]))

    random.shuffle(temp_data)

    x, y = [], []
    for xy_i in temp_data:
        x_i, y_i = xy_i
        x.append(x_i)
        y.append(y_i)

    b_src = []
    for j in range(data_len):
        if len(b_src) == batch_size:

            seqlen_cache = [len(w[0]) for w in b_src]
            x_cache = [w[0] for w in b_src]
            y_cache = [w[1] for w in b_src]

            yield (x_cache, y_cache, seqlen_cache)

            b_src = []

        b_src.append((x[j], y[j]))

    if len(b_src) == batch_size:

        seqlen_cache = [len(w[0]) for w in b_src]
        x_cache = [w[0] for w in b_src]
        y_cache = [w[1] for w in b_src]

        yield (x_cache, y_cache, seqlen_cache)
   
def generate_movies(n_samples=1200, n_frames=15):
    row = 80
    col = 80
    noisy_movies = np.zeros((n_samples, n_frames, row, col, 1), dtype=np.float)
    shifted_movies = np.zeros((n_samples, n_frames, row, col, 1),
                              dtype=np.float)

    for i in range(n_samples):
        # Add 3 to 7 moving squares
        n = np.random.randint(3, 7)

        for j in range(n):
            # Initial position
            xstart = np.random.randint(20, 60)
            ystart = np.random.randint(20, 60)
            # Direction of motion
            directionx = np.random.randint(0, 3) - 1
            directiony = np.random.randint(0, 3) - 1

            # Size of the square
            w = np.random.randint(2, 4)

            for t in range(n_frames):
                x_shift = xstart + directionx * t
                y_shift = ystart + directiony * t
                noisy_movies[i, t, x_shift - w: x_shift + w,
                             y_shift - w: y_shift + w, 0] += 1

                # Make it more robust by adding noise.
                # The idea is that if during inference,
                # the value of the pixel is not exactly one,
                # we need to train the network to be robust and still
                # consider it as a pixel belonging to a square.
                if np.random.randint(0, 2):
                    noise_f = (-1)**np.random.randint(0, 2)
                    noisy_movies[i, t,
                                 x_shift - w - 1: x_shift + w + 1,
                                 y_shift - w - 1: y_shift + w + 1,
                                 0] += noise_f * 0.1

                # Shift the ground truth by 1
                x_shift = xstart + directionx * (t + 1)
                y_shift = ystart + directiony * (t + 1)
                shifted_movies[i, t, x_shift - w: x_shift + w,
                               y_shift - w: y_shift + w, 0] += 1


    # Cut to a 40x40 window
    noisy_movies = noisy_movies[::, ::, 20:60, 20:60, ::]
    shifted_movies = shifted_movies[::, ::, 20:60, 20:60, ::]
    noisy_movies[noisy_movies >= 1] = 1
    shifted_movies[shifted_movies >= 1] = 1
    return noisy_movies, shifted_movies

    