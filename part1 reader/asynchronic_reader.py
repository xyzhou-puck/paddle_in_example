#encoding=utf8

import os
import sys
import numpy as np
import paddle
import paddle.fluid as fluid

def create_sample_reader(true_factors):
    def sample_reader():
        while True:
            x_sample = np.random.randint(5, size = (1, 4))
            y_sample = np.array(
                    true_factors[0] * x_sample[0][0] + true_factors[1] * x_sample[0][1] + true_factors[2] * x_sample[0][2] + true_factors[3] * x_sample[0][3]).reshape((1,1))
            #y_sample[0,0] = true_factors[0] * x_sample[0] + true_factors[1] * x_sample[1] + true_factors[2] * x_sample[2] + true_factors[3] * x_sample[3]

            yield [(x_sample, y_sample)]

    return sample_reader


def create_batch_reader(true_factors, batch_size = 100):
    def batch_reader():
        batch = []
        while True:
            x_sample = np.random.randint(5, size = (1, 4))
            y_sample = np.array(
                true_factors[0] * x_sample[0][0] + true_factors[1] * x_sample[0][1] + true_factors[2] * x_sample[0][2] + true_factors[3] * x_sample[0][3]).reshape((1,1))

            batch.append((x_sample, y_sample))

            if len(batch) == batch_size:
                yield batch
                batch = []
            
    return batch_reader

def asynchronic_reader_creator(shape, dtype, capacity, lod_level, name = "asyn_reader"):

    reader = fluid.layers.py_reader(
        capacity = capacity,
        shapes = shape,
        dtypes = dtype,
        lod_levels = lod_level,
        use_double_buffer = True,
        name = "asyn_reader")

    return reader


if __name__ == "__main__":

    reader = asynchronic_reader_creator(((10, 4), (10, 1)), ("float32", "float32"), 100, "asy_reader")


