from __future__ import print_function
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Conv2DTranspose, Pool2D, FC, BatchNorm
from paddle.fluid.dygraph.base import to_variable
from PIL import Image
import cv2
import time
import os
import vgg

img_shape = (3, 128, 128)
content_path = 'test.jpeg'
style_path = 'pattern.jpg'

num_iterations = 2000
content_weight = 0.005
style_weight = 0.8
total_variation_weight = 0.02

def image_load(image_file, target_size):
    img = cv2.imread(image_file)
    img = cv2.resize(img, target_size)
    img = np.array(img).astype('float32').transpose((2, 0, 1))
    img = img / 255
    return img

def process_image(img):
    img -= np.array(vgg.train_parameters['input_mean']).reshape((3, 1, 1))
    img /= np.array(vgg.train_parameters['input_std']).reshape((3, 1, 1))
    img = np.expand_dims(img, axis=0)
    return img

def deprocess_image(img):
    img *= np.array(vgg.train_parameters['input_std']).reshape((3, 1, 1))
    img += np.array(vgg.train_parameters['input_mean']).reshape((3, 1, 1))
    img = img * 255.0
    img = np.clip(img, 0, 255).astype('uint8')
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img, 'RGB')
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    img = np.array(img).astype('uint8')
    return img

def load_and_process_image(image_file):
    img = image_load(image_file, (128, 128))
    return process_image(img)

content_image = image_load(content_path, (128, 128))
style_image = image_load(style_path, (128, 128))
content_layers = ['block5_conv2']
style_layers = ['block1_conv1', 
                  'block2_conv1',
                  'block3_conv1', 
                  'block4_conv1',
                  'block5_conv1']
num_content_layers = len(content_layers)
num_style_layers = len(style_layers)

def content_loss(content_image, target):
    return fluid.layers.reduce_sum(fluid.layers.square(target - content_image))

def gram_matrix(x):
    features = fluid.layers.reshape(x, [x.shape[0], -1])
    return fluid.layers.matmul(features, features, False, True)

def style_loss(gram_target, combination):
    gram_comb = gram_matrix(combination)
    dnmtr = (4.0 * (img_shape[0] ** 2) * ((img_shape[1] * img_shape[2]) ** 2))
    return fluid.layers.reduce_sum(fluid.layers.square(gram_target - gram_comb)) / dnmtr

def total_variation_loss(image):
    y_ij  = image[:, :,  :img_shape[1] - 1,  :img_shape[2] - 1]
    y_i1j = image[:, :, 1:,                  :img_shape[2] - 1]
    y_ij1 = image[:, :,  :img_shape[1] - 1, 1:]

    a = fluid.layers.square(y_ij - y_i1j)
    b = fluid.layers.square(y_ij - y_ij1)
    return fluid.layers.reduce_sum(fluid.layers.pow(a + b, 1.25))

def get_feature_representations(model, content_path, style_path):
    # dim == (1, 3, 128, 128)
    style_image_np = load_and_process_image(style_path)
    # dim == (1, 3, 128, 128)
    content_image_np = load_and_process_image(content_path)
    # dim == (2, 3, 128, 128)
    stack_images_np = np.concatenate([style_image_np, content_image_np])

    # len(model_outputs) == 6
    stack_images = to_variable(stack_images_np)
    model_outputs = model(stack_images)
    
    style_features = [model_outputs[layer_name][0] for layer_name in style_layers]
    content_features = [model_outputs[layer_name][1] for layer_name in content_layers]

    return style_features, content_features

def compute_loss(model, loss_weights, init_image, gram_style_features, content_features):
    style_weight, content_weight = loss_weights
    
    model_outputs = model(init_image)

    style_output_features = [model_outputs[layer_name] for layer_name in style_layers]
    content_output_features = [model_outputs[layer_name] for layer_name in content_layers]

    weight_per_style_layer = 1.0 / num_style_layers
    style_score = np.zeros((1,)).astype('float32')
    style_score = to_variable(style_score)

    for l in range(len(style_layers)):
        target_style = gram_style_features[l]
        comb_style = style_output_features[l]
        score = weight_per_style_layer * style_loss(target_style, comb_style[0])
        style_score += score

    weight_per_content_layer = 1.0 / num_content_layers
    content_score = np.zeros((1,)).astype('float32')
    content_score = to_variable(content_score)

    for l in range(len(content_layers)):
        target_content = content_features[l]
        comb_style = content_output_features[l]
        score = weight_per_content_layer * content_loss(comb_style[0], target_content)
        content_score += score

    variation_loss = total_variation_loss(init_image[0])
    style_score *= style_weight
    content_score *= content_weight
    variation_score = variation_loss * total_variation_weight

    loss = style_score + content_score + variation_score

    return loss, style_score, content_score, variation_score

def compute_grads(model, loss_weights, init_image, gram_style_features, content_features):

    loss, _, _, _ = compute_loss(
        model, 
        loss_weights, 
        init_image, 
        gram_style_features, 
        content_features)

    loss.backward()
    grads = init_image.gradient()

    return grads, loss

def run_style_transfer():
    place = fluid.CUDAPlace(1)
    with fluid.dygraph.guard(place):

        vgg19 = vgg.build_vgg19('vgg19')

        if os.path.exists('./pretrained'):
            print('load pretrained model...')
            model, _ = fluid.dygraph.load_persistables('./pretrained')
            vgg19.load_dict(model)
        else:
            print('cannot load pretrained model!')
            exit(0)

        style_features, content_features = get_feature_representations(
            vgg19, content_path, style_path)

        gram_style_features = []
        for l in range(len(style_features)):
            gram_style_features.append(gram_matrix(style_features[l]))

        init_image_np = load_and_process_image(content_path)
        init_image = to_variable(init_image_np)

        best_image_np = np.zeros_like(init_image_np).astype('float32')
        best_image = to_variable(best_image_np)

        best_loss = np.inf

        loss_weights = style_weight, content_weight

        global_start = time.time()

        lr = 1e-5
        for i in range(num_iterations):
            grads, loss = compute_grads(vgg19,
                                        loss_weights,
                                        init_image,
                                        gram_style_features,
                                        content_features)

            init_image = init_image - lr * to_variable(grads)

            init_image = fluid.layers.reshape(init_image, [3, 128, 128])
            init_image_0 = fluid.layers.clip(init_image[0], -2.118, 2.249)
            init_image_1 = fluid.layers.clip(init_image[1], -2.036, 2.429)
            init_image_2 = fluid.layers.clip(init_image[2], -1.804, 2.640)
            init_image = fluid.layers.concat([init_image_0, init_image_1, init_image_2])
            init_image = fluid.layers.reshape(init_image, [1, 3, 128, 128])

            if loss.numpy()[0] < best_loss:
                best_loss = loss.numpy()[0]
                best_image = init_image

            if i % 50 == 0:
                print('step: %d, total loss: %f' % (i, loss.numpy()[0]))

                if i % 100 == 0:
                    fname = 'style_epoch_%d.jpeg' % i
                    output_img = fluid.layers.reshape(best_image, [3, 128, 128]).numpy()
                    output_img = deprocess_image(output_img)
                    output_img = Image.fromarray(output_img, 'RGB')
                    output_img.save(fname, format='JPEG')

        print('total time: %ds' % (time.time() - global_start))

if __name__ == '__main__':
    run_style_transfer()
