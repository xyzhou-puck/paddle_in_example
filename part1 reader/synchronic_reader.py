#encoding=utf8

import os
import sys
import numpy as np
import paddle
import paddle.fluid as fluid

def sample_reader_creator(true_factors):
    def sample_reader():
        while True:
            x_sample = np.random.randint(5, size = (4))
            y_sample = true_factors[0] * x_sample[0] + true_factors[1] * x_sample[1] + true_factors[2] * x_sample[2] + true_factors[3] * x_sample[3]

            yield ([x_sample.tolist()], [y_sample])

    return sample_reader


def batch_reader_creator(true_factors, batch_size):

    def batch_reader():
        batch_x = []
        batch_y = []
        while True:
            x_sample = np.random.randint(5, size = (4))
            y_sample = true_factors[0] * x_sample[0] + true_factors[1] * x_sample[1] + true_factors[2] * x_sample[2] + true_factors[3] * x_sample[3]

            batch_x.append(x_sample.tolist())
            batch_y.append([y_sample]) # well, this is stupid

            if len(batch_x) == batch_size:
                yield (batch_x, batch_y)
                batch_x = []
                batch_y = []

    return batch_reader


if __name__ == "__main__":
    
    batch_reader = batch_reader_creator([1,2,3,4], 10)

    for next_batch in batch_reader():
        print(next_batch)

