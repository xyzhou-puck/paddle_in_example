'''Neural style transfer with Paddle.

Run the script with:
```
python neural_style_transfer.py path_to_your_base_image.jpg \
    path_to_your_reference.jpg prefix_for_results
```
e.g.:
```
python neural_style_transfer.py img/tuebingen.jpg \
    img/starry_night.jpg results/my_result
```
Optional parameters:
```
--iter, To specify the number of iterations \
    the style transfer takes place (Default is 10)
--content_weight, The weight given to the content loss (Default is 0.025)
--style_weight, The weight given to the style loss (Default is 1.0)
--tv_weight, The weight given to the total variation loss (Default is 1.0)
```

It is preferable to run this script on GPU, for speed.

Example result: https://twitter.com/fchollet/status/686631033085677568

# Details

Style transfer consists in generating an image
with the same "content" as a base image, but with the
"style" of a different picture (typically artistic).

This is achieved through the optimization of a loss function
that has 3 components: "style loss", "content loss",
and "total variation loss":

- The total variation loss imposes local spatial continuity between
the pixels of the combination image, giving it visual coherence.

- The style loss is where the deep learning keeps in --that one is defined
using a deep convolutional neural network. Precisely, it consists in a sum of
L2 distances between the Gram matrices of the representations of
the base image and the style reference image, extracted from
different layers of a convnet (trained on ImageNet). The general idea
is to capture color/texture information at different spatial
scales (fairly large scales --defined by the depth of the layer considered).

 - The content loss is a L2 distance between the features of the base
image (extracted from a deep layer) and the features of the combination image,
keeping the generated image close enough to the original one.

# References
    - [A Neural Algorithm of Artistic Style](http://arxiv.org/abs/1508.06576)
'''

from __future__ import print_function
import numpy as np
from scipy.optimize import fmin_l_bfgs_b
import time
import argparse
from PIL import Image, ImageEnhance
import vgg
import paddle
import paddle.fluid as fluid
import cv2
from paddle.fluid.backward import calc_gradient

PRETRAINED_MODEL = "./pretrained"

parser = argparse.ArgumentParser(description='Neural style transfer with Paddle.')
parser.add_argument('base_image_path', metavar='base', type=str,
                    help='Path to the image to transform.')
parser.add_argument('style_reference_image_path', metavar='ref', type=str,
                    help='Path to the style reference image.')
parser.add_argument('result_prefix', metavar='res_prefix', type=str,
                    help='Prefix for the saved results.')
parser.add_argument('--iter', type=int, default=10, required=False,
                    help='Number of iterations to run.')
parser.add_argument('--content_weight', type=float, default=0.005, required=False,
                    help='Content weight.')
parser.add_argument('--style_weight', type=float, default=0.8, required=False,
                    help='Style weight.')
parser.add_argument('--tv_weight', type=float, default=0.02, required=False,
                    help='Total Variation weight.')

args = parser.parse_args()
base_image_path = args.base_image_path
style_reference_image_path = args.style_reference_image_path
result_prefix = args.result_prefix
iterations = args.iter

# these are the weights of the different loss components
total_variation_weight = args.tv_weight
style_weight = args.style_weight
content_weight = args.content_weight

# dimensions of the generated picture.
width, height = Image.open(base_image_path).size
img_nrows = 128
img_ncols = int(width * img_nrows / height)

# util function to open, resize and format pictures into appropriate tensors

def preprocess_image(image_path):

    img = cv2.imread(image_path)
    img = cv2.resize(img, (img_ncols, img_nrows))
    img = np.array(img).astype('float32').transpose((2, 0, 1)) / 255.0
    img -= np.array(vgg.train_parameters['input_mean']).reshape((3, 1, 1))
    img /= np.array(vgg.train_parameters['input_std']).reshape((3, 1, 1))
    img = np.expand_dims(img, axis=0)
    return img

# util function to convert a tensor into a valid image


def deprocess_image(img):
    # Remove zero-center by mean pixel
    img = np.array(img).astype('float32').reshape((3, img_nrows, img_ncols))
    img *= np.array(vgg.train_parameters['input_std']).reshape((3, 1, 1))
    img += np.array(vgg.train_parameters['input_mean']).reshape((3, 1, 1))
    img = np.transpose(img, (1, 2, 0)) * 255.0
    img = np.clip(img, 0, 255).astype('uint8')
    img = Image.fromarray(img, 'RGB')
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    return img

# get tensor representations of our images
base_image = preprocess_image(base_image_path)
style_reference_image = preprocess_image(style_reference_image_path)

# this will contain our generated image
combination_image = np.zeros((1, 3, img_nrows, img_ncols))


base_data = fluid.layers.data(
    name='base_image', 
    shape=(3, img_nrows, img_ncols), 
    dtype='float32')
style_reference_data = fluid.layers.data(
    name='style_reference_image', 
    shape=(3, img_nrows, img_ncols), 
    dtype='float32')
combination_data = fluid.layers.data(
    name='combination_image', 
    shape=(3, img_nrows, img_ncols), 
    dtype='float32', 
    stop_gradient=False)

# combine the 3 images into a single tensor
input_tensor = fluid.layers.concat([base_data, style_reference_data, combination_data])

# build the VGG19 network with our 3 images as input
# the model will be loaded with pre-trained weights
model = vgg.VGG19()
outputs_dict = model.net(input=input_tensor)

# compute the neural style loss
# first we need to define 4 util functions

# the gram matrix of an image tensor (feature-wise outer product)

def gram_matrix(x):
    assert len(x.shape) == 3

    features = fluid.layers.reshape(x, (-1, x.shape[0], x.shape[1] * x.shape[2]))
    gram = fluid.layers.matmul(features, features, False, True)
    gram = fluid.layers.squeeze(gram, [0])
    return gram

# the "style loss" is designed to maintain
# the style of the reference image in the generated image.
# It is based on the gram matrices (which capture style) of
# feature maps from the style reference image
# and from the generated image

def style_loss(style, combination):
    assert len(style.shape) == 3
    assert len(combination.shape) == 3

    S = gram_matrix(style)
    C = gram_matrix(combination)

    channels = 3
    size = img_nrows * img_ncols
    dnmtr = (4.0 * (channels ** 2) * (size ** 2))
    return fluid.layers.reduce_sum(fluid.layers.square(S - C)) / dnmtr

# an auxiliary loss function
# designed to maintain the "content" of the
# base image in the generated image

def content_loss(base, combination):
    assert len(base.shape) == 3
    assert len(combination.shape) == 3

    return fluid.layers.reduce_sum(fluid.layers.square(combination - base))

# the 3rd loss function, total variation loss,
# designed to keep the generated image locally coherent

def total_variation_loss(x):
    assert len(x.shape) == 4

    a = x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, 1:, :img_ncols - 1]
    b = x[:, :, :img_nrows - 1, :img_ncols - 1] - x[:, :, :img_nrows - 1, 1:]
    a = fluid.layers.square(a)
    b = fluid.layers.square(b)

    return fluid.layers.reduce_sum(fluid.layers.pow(a + b, 1.25))

# combine these loss functions into a single scalar
layer_features = outputs_dict['block5_conv2']

base_image_features = layer_features[0]
combination_features = layer_features[2]
loss = content_weight * content_loss(base_image_features, combination_features)
feature_layers = ['block1_conv1', 'block2_conv1',
                  'block3_conv1', 'block4_conv1',
                  'block5_conv1']

for layer_name in feature_layers:
    layer_features = outputs_dict[layer_name]
    style_reference_features = layer_features[1]
    combination_features = layer_features[2]
    sl = style_loss(style_reference_features, combination_features)
    loss += (style_weight / len(feature_layers)) * sl
loss += total_variation_weight * total_variation_loss(combination_data)

# get the gradients of the generated image wrt the loss
grads = calc_gradient(loss, combination_data)

fetch = [loss.name]
if isinstance(grads, (list, tuple)):
    fetch.append(grads[0].name)
else:
    fetch.append(grads.name)

optimizer = fluid.optimizer.SGD(0.0)
optimizer.backward(loss=loss)

test_program = fluid.default_main_program()

exe = fluid.Executor(fluid.CUDAPlace(1))
exe.run(fluid.default_startup_program())

fluid.io.load_persistables(exe, PRETRAINED_MODEL)
print('Model loaded.')

def inference(base, reference, combination):
    input_feed = {}
    input_feed['base_image'] = np.array(base).astype("float32")
    input_feed['style_reference_image'] = np.array(reference).astype("float32")
    input_feed['combination_image'] = np.array(combination).astype("float32")

    result = exe.run(test_program, fetch_list=fetch, feed=input_feed)

    return result

def eval_loss_and_grads(x):
    base = base_image.reshape((1, 3, img_nrows, img_ncols))
    reference = style_reference_image.reshape((1, 3, img_nrows, img_ncols))
    x = x.reshape((1, 3, img_nrows, img_ncols))
    outs = inference(base, reference, x)
    loss_value = outs[0]
    if len(outs[1:]) == 1:
        grad_values = outs[1].flatten().astype('float64')
    else:
        grad_values = np.array(outs[1:]).flatten().astype('float64')
    return loss_value, grad_values

# this Evaluator class makes it possible
# to compute loss and gradients in one pass
# while retrieving them via two separate functions,
# "loss" and "grads". This is done because scipy.optimize
# requires separate functions for loss and gradients,
# but computing them separately would be inefficient.

class Evaluator(object):

    def __init__(self):
        self.loss_value = None
        self.grads_values = None

    def loss(self, x):
        assert self.loss_value is None
        loss_value, grad_values = eval_loss_and_grads(x)
        self.loss_value = loss_value
        self.grad_values = grad_values
        return self.loss_value

    def grads(self, x):
        assert self.loss_value is not None
        grad_values = np.copy(self.grad_values)
        self.loss_value = None
        self.grad_values = None
        return grad_values

evaluator = Evaluator()

# run scipy-based optimization (L-BFGS) over the pixels of the generated image
# so as to minimize the neural style loss
x = preprocess_image(base_image_path)

for i in range(iterations):
    print('Start of iteration', i)
    start_time = time.time()

    x, min_val, info = fmin_l_bfgs_b(evaluator.loss, 
                                     x0=x.flatten().astype('float64'),
                                     fprime=evaluator.grads, 
                                     maxfun=20)

    print('Current loss value:', min_val)
    # save current generated image
    img = deprocess_image(x.copy())
    fname = result_prefix + '_at_iteration_%d.jpeg' % i
    img.save(fname, format='JPEG')
    end_time = time.time()
    print('Image saved as', fname)
    print('Iteration %d completed in %ds\n' % (i, end_time - start_time))

