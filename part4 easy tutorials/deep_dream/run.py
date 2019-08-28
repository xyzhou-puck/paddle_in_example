from __future__ import print_function

import os
import time
import sys
import math
import numpy as np
import argparse
import functools
import scipy
import scipy.ndimage
from matplotlib import cm
from PIL import Image

import reader

import paddle
import paddle.fluid as fluid
from inception_v4 import InceptionV4
from paddle.fluid.backward import calc_gradient

PRETRAINED_MODEL = "./pretrained"
IMAGE_SHAPE = (3,224,224)

image = fluid.layers.data(name='image', shape=IMAGE_SHAPE, dtype='float32', stop_gradient=False)

model = InceptionV4()
loss, grad, _ = model.net(input=image)

optimizer = fluid.optimizer.SGD(0.0)
optimizer.backward(loss=loss)

test_program = fluid.default_main_program()

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

fluid.io.load_params(exe, PRETRAINED_MODEL)
print('Model loaded.')

fetch_list = [loss.name, grad.name]

def resize_img(img, size):
    img = np.copy(img)
    factors = (1, 1,
               float(size[0]) / img.shape[2],
               float(size[1]) / img.shape[3])

    return scipy.ndimage.zoom(img, factors, order=1)

def prepare_input(batch):
    x, _ = batch
    res = {}
    res['image'] = np.array(x).astype("float32")

    return res

def fetch_loss_grad(x):
    data_iter = reader.get_data_iter(x, 1)
    for batch_id, batch in enumerate(data_iter):
        input_data_feed = prepare_input(batch)

        result = exe.run(test_program,
                         fetch_list=fetch_list,
                         feed=input_data_feed)

        loss = result[0]
        grad = result[1]

        return loss, grad

def gradient_ascent(x, iterations, step, max_loss=None):
    for i in range(iterations):
        loss_value, grad_values = fetch_loss_grad(x)
        if max_loss is not None and loss_value > max_loss:
            break
        print('..Loss value at', i, ':', loss_value)
        x += step * grad_values
    return x

step = 0.1  # Gradient ascent step size
num_octave = 3  # Number of scales at which to run gradient ascent
octave_scale = 1.4  # Size ratio between scales
iterations = 10  # Number of ascent steps per scale
max_loss = 0.1

# load original image
img, original_shape = reader.process_image('./data/fishes.jpg')

print(original_shape)

successive_shapes = [original_shape]
for i in range(1, num_octave):
    shape = tuple([int(dim / (octave_scale ** i)) for dim in original_shape])
    successive_shapes.append(shape)
    print(shape)

successive_shapes = successive_shapes[::-1]
original_img = np.copy(img)
shrunk_original_img = resize_img(img, successive_shapes[0])

# gradient ascent
for shape in successive_shapes:
    print('Processing image shape', shape)
    img = resize_img(img, shape)
    img = gradient_ascent(img,
                          iterations=iterations,
                          step=step,
                          max_loss=max_loss)
    upscaled_shrunk_original_img = resize_img(shrunk_original_img, shape)
    same_size_original = resize_img(original_img, shape)
    lost_detail = same_size_original - upscaled_shrunk_original_img

    img += lost_detail
    shrunk_original_img = resize_img(original_img, shape)

# save result
im = reader.deprocess_image(img[0])
im.save('./data/test.jpeg', format='JPEG')


























