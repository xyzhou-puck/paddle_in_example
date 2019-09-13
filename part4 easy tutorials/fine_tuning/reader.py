import os
import math
import random
import functools
import numpy as np
import cv2
import io
import signal
from PIL import Image
from progress.bar import Bar

import paddle
import paddle.fluid as fluid

random.seed(0)
np.random.seed(0)

def rotate_image(img, deg):
    (h, w) = img.shape[:2]
    center = (w / 2, h / 2)
    angle = np.random.randint(-deg, deg)
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    rotated = cv2.warpAffine(img, M, (w, h))
    return rotated

def crop_image(img, h_size, w_size, center=False):
    height, width = img.shape[:2]
    if center == True:
        w_start = (width - w_size) // 2
        h_start = (height - h_size) // 2
    else:
        w_start = np.random.randint(0, width - w_size + 1)
        h_start = np.random.randint(0, height - h_size + 1)
    w_end = w_start + w_size
    h_end = h_start + h_size
    img = img[h_start:h_end, w_start:w_end, :]
    return img

def random_crop(img, width_shift, height_shift):
    if width_shift == 0 and height_shift == 0:
        return img
    (h, w) = img.shape[:2]
    target_width = int(w - width_shift * w * 1.0);
    target_height = int(h - height_shift * h * 1.0);
    return crop_image(img, target_height, target_width)

def resize_short(img, target_size):
    percent = float(target_size) / min(img.shape[0], img.shape[1])
    resized_width = int(round(img.shape[1] * percent))
    resized_height = int(round(img.shape[0] * percent))
    resized = cv2.resize(img, (resized_width, resized_height))
    return resized

def process_image(sample, mode='test', target_size=0, rotate=0, w_shift=0, h_shift=0, flip=False):

    img_path = sample[0]
    img = cv2.imread(img_path)

    if mode == 'train':
        if rotate > 0:
            img = rotate_image(img, rotate)
        if w_shift > 0 or h_shift > 0:
            img = random_crop(img, w_shift, h_shift)
        if flip and np.random.randint(0, 2) == 1:
            img = img[:, ::-1, :]
        if target_size > 0:
            img = resize_short(img, target_size)
            img = crop_image(img, target_size, target_size, center=True)
    else:
        if target_size > 0:
            img = resize_short(img, target_size)
            img = crop_image(img, target_size, target_size, center=True)

    img = img[:, :, ::-1].astype('float32').transpose((2, 0, 1)) / 255

    if mode == 'train' or mode == 'val':
        return (img, np.array([sample[1]]).astype('int64'))
    elif mode == 'test':
        return (img, np.array([0]).astype('int64'))

def raw_data(data_dir, mode, target_size, rotate=0, w_shift=0, h_shift=0, flip=False):
    x, y = [], []
    for path, subdirs, files in os.walk(data_dir):
        for filename in files:
            if filename.endswith('.jpg'):
                img_path = os.path.join(path, filename)
                if not os.path.exists(img_path):
                    print("Warning: {} doesn't exist!".format(img_path))
                sample = []
                sample.append(img_path)
                sample.append((img_path.find('Cat') < 0) and 0 or 1)
                sx, sy = process_image(sample, mode, target_size, 
                    rotate, w_shift, h_shift, flip)
                x.append(sx)
                y.append(sy)
    return (x, y)

def reader_creator(data, mode='train'):
    def reader():
        (inputs, labels) = data
        for input, label in zip(inputs, labels):
            if mode == "train" or mode == "val":
                yield input, label
            elif mode == "test":
                yield input, label
    return reader

def train(data_dir, target_size, rotate, w_shift, h_shift, flip):
    return raw_data(
        data_dir,
        'train',
        target_size=target_size,
        rotate=rotate,
        w_shift=w_shift,
        h_shift=h_shift,
        flip=flip)


def val(data_dir, target_size):
    return raw_data(data_dir, 'val', target_size=target_size)


def test(data_dir, target_size):
    return raw_data(data_dir, 'test', target_size=target_size)