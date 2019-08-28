import os
import math
import random
import functools
import numpy as np
from PIL import Image, ImageEnhance

import paddle

random.seed(0)
np.random.seed(0)

DATA_DIM = 224

THREAD = 8
BUF_SIZE = 2048

DATA_DIR = './data'

img_mean = np.array([0.485, 0.456, 0.406]).reshape((3, 1, 1))
img_std = np.array([0.229, 0.224, 0.225]).reshape((3, 1, 1))

def deprocess_image(sample):
    sample_img = np.array(sample).astype('float32')
    sample_img *= img_std
    sample_img += img_mean
    img_temp = np.array(sample_img).transpose(2, 1, 0) * 255
    img_temp = np.clip(img_temp, 0, 255)
    return Image.fromarray(np.array(img_temp).astype('uint8'), 'RGB')

def process_image(sample):
    img_path = sample

    print('Image loaded.')

    img = Image.open(img_path)
    size = img.size

    if img.mode != 'RGB':
        img = img.convert('RGB')

    img = np.array(img).astype('float32').transpose((2, 1, 0)) / 255
    img -= img_mean
    img /= img_std

    return [img], size

def get_data_iter(raw_data, batch_size):
    x = raw_data
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

        b_src.append((x[index[j]], 0))
    if len(b_src) == batch_size * cache_num:

        new_cache = b_src

        for i in range(cache_num):
            batch_data = new_cache[i * batch_size:(i + 1) * batch_size]
            x_cache = [w[0] for w in batch_data]
            y_cache = [[w[1]] for w in batch_data]
            yield (x_cache, y_cache)