"""
Train an Auxiliary Classifier GAN (ACGAN) on the MNIST dataset.

[More details on Auxiliary Classifier GANs.](https://arxiv.org/abs/1610.09585)

You should start to see reasonable images after ~5 epochs, and good images
by ~15 epochs. You should use a GPU, as the convolution-heavy operations are
very slow on the CPU. 

Timings:

Hardware           | Time / Epoch
:------------------|------------:
 Xeon Sliver 4114  |   ~30 min
 GTX 1080 (pascal) |   ~80 sec 

Consult [Auxiliary Classifier Generative Adversarial Networks in Keras
](https://github.com/lukedeo/keras-acgan) for more information and example output.
"""

from __future__ import print_function
from collections import defaultdict
try:
    import cPickle as pickle
except ImportError:
    import pickle

import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from base_network import norm_layer, deconv2d, linear, conv2d
from paddle.dataset import mnist
from progress.bar import Bar
from PIL import Image
from itertools import islice

# np.random.seed(1337)
num_classes = 10

class ACGAN(object):
    def __init__(self, latent_size, num_classes, batch_size_one=False):
        self.latent_size = latent_size
        self.num_classes = num_classes
        self.batch_size_one = batch_size_one
        if self.batch_size_one:
            self.norm = None
        else:
            self.norm = 'batch_norm'

    def network_g(self, image_class, latent, name="generator"):
        cls = layers.embedding(
            image_class, 
            size=[self.num_classes, self.latent_size],
            param_attr=fluid.ParamAttr(
                name=name + '_emb',
                initializer=fluid.initializer.Xavier()))

        h = layers.elementwise_mul(latent, cls)

        h_fc = linear(input=h, output_size=3 * 3 * 384, 
            name=name + '_fc', activation_fn='leaky_relu')
        h_reshape = layers.reshape(h_fc, (-1, 384, 3, 3))
        h_deconv1 = deconv2d(h_reshape, num_filters=192, filter_size=5, 
            stride=1, padding_type="VALID", activation_fn='leaky_relu', 
            norm=self.norm, name=name + '_deconv1')
        h_deconv2 = deconv2d(h_deconv1, num_filters=96, filter_size=5, 
            stride=2, padding=1, activation_fn='leaky_relu', 
            norm=self.norm, name=name + '_deconv2')
        h_deconv3 = deconv2d(h_deconv2, num_filters=1, filter_size=5, 
            stride=2, padding=2, activation_fn='tanh',
            name=name + '_deconv3')

        h_deconv3 = layers.crop(h_deconv3, shape=(-1, 1, 28, 28))

        return h_deconv3

    def network_d(self, input, name="discriminator"):

        h_conv1 = conv2d(input, num_filters=64, filter_size=5, stride=2,
            padding=2, activation_fn='leaky_relu', 
            relufactor=0.2, name=name + '_conv1')
        h_dropout1 = layers.dropout(h_conv1, dropout_prob=0.3)

        h_conv2 = conv2d(h_dropout1, num_filters=128, filter_size=5, stride=2,
            padding=2, activation_fn='leaky_relu', 
            relufactor=0.2, name=name + '_conv2')
        h_dropout2 = layers.dropout(h_conv2, dropout_prob=0.3)

        h_conv3 = conv2d(h_dropout2, num_filters=128, filter_size=3, stride=2,
            padding_type="SAME", activation_fn='leaky_relu', 
            relufactor=0.2, name=name + '_conv3')
        h_dropout3 = layers.dropout(h_conv3, dropout_prob=0.3)

        h_conv4 = conv2d(h_dropout3, num_filters=256, filter_size=3, stride=1,
            padding_type="SAME", activation_fn='leaky_relu', 
            relufactor=0.2, name=name + '_conv4')
        h_dropout4 = layers.dropout(h_conv4, dropout_prob=0.3)

        features = layers.flatten(h_conv2)

        fake = linear(features, output_size=1, name=name + '_fc1')
        aux = linear(features, output_size=self.num_classes, name=name + '_fc2')

        return fake, aux

class DTrain():
    def __init__(self, x, y, y_aux, cfg):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = ACGAN(cfg.latent_size, cfg.num_classes)
            self.fake, self.aux = model.network_d(x, name='d')

            self.fake_loss = layers.sigmoid_cross_entropy_with_logits(
                x=self.fake, label=y)
            self.aux_loss = layers.softmax_with_cross_entropy(
                logits=self.aux, label=y_aux)
            self.unweighted_loss = layers.reduce_sum(self.fake_loss + self.aux_loss)
            self.infer_program = self.program.clone(for_test=True)

            # we don't want the discriminator to also maximize the classification
            # accuracy of the auxiliary classifier on generated images, so we
            # don't train discriminator to produce class labels for generated
            # images (see https://openreview.net/forum?id=rJXTf9Bxg).
            # To preserve sum of sample weights for the auxiliary classifier,
            # we assign sample weight of 2 to the real images.

            fake_loss_weight = layers.ones(shape=[cfg.batch_size * 2, 1], dtype='float32')
            aux_loss_weight_zeros = layers.zeros(shape=[cfg.batch_size, 1], dtype='float32')
            aux_loss_weight_twos = layers.fill_constant(
                shape=[cfg.batch_size, 1], value=2.0, dtype='float32')
            aux_loss_weight = layers.concat([aux_loss_weight_twos, aux_loss_weight_zeros])

            self.fake_loss = layers.elementwise_mul(self.fake_loss, fake_loss_weight)
            self.aux_loss = layers.elementwise_mul(self.aux_loss, aux_loss_weight)

            self.loss = layers.reduce_sum(self.fake_loss) + layers.reduce_sum(self.aux_loss)

            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and (var.name.startswith("d")):
                    vars.append(var.name)
            optimizer = fluid.optimizer.Adam(
                learning_rate=cfg.adam_lr, beta1=cfg.adam_beta_1, name="net_d")
            optimizer.minimize(self.loss, parameter_list=vars)

class GTrain():
    def __init__(self, sampled_labels, noise, trick, cfg):
        self.program = fluid.default_main_program().clone()
        with fluid.program_guard(self.program):
            model = ACGAN(cfg.latent_size, cfg.num_classes)
            self.fake_img = model.network_g(sampled_labels, noise, name='g')
            self.infer_program = self.program.clone(for_test=True)
            self.fake, self.aux = model.network_d(self.fake_img, name="d")
            self.fake_loss = layers.reduce_sum(layers.sigmoid_cross_entropy_with_logits(
                x=self.fake, label=trick))
            self.aux_loss = layers.reduce_sum(layers.softmax_with_cross_entropy(
                logits=self.aux, label=sampled_labels))
            self.loss = self.fake_loss + self.aux_loss

            vars = []
            for var in self.program.list_vars():
                if fluid.io.is_parameter(var) and (var.name.startswith("g")):
                    vars.append(var.name)
            optimizer = fluid.optimizer.Adam(
                learning_rate=cfg.adam_lr, beta1=cfg.adam_beta_1, name="net_g")
            optimizer.minimize(self.loss, parameter_list=vars)

class MnistACGAN(object):
    def __init__(self, cfg):
        self.cfg = cfg

    def build_model(self):
        image_batch = layers.data(name='image_batch', shape=[-1, 1, 28, 28],
            dtype='float32')
        label_batch = layers.data(name='label_batch', shape=[-1, 1],
            dtype='int64')
        noise = layers.data(name='noise', shape=[-1, self.cfg.latent_size],
            dtype='float32')
        sampled_labels = layers.data(name='sampled_labels', shape=[-1, 1],
            dtype='int64')
        x = layers.data(name='x', shape=[-1, 1, 28, 28], dtype='float32')
        y = layers.data(name='y', shape=[-1, 1], dtype='float32')
        aux_y = layers.data(name='aux_y', shape=[-1, 1], dtype='int64')
        trick = layers.data(name='trick', shape=[-1, 1], dtype='float32')

        g_train = GTrain(sampled_labels, noise, trick, self.cfg)
        d_train = DTrain(x, y, aux_y, self.cfg)

        place = fluid.CPUPlace()
        exe = fluid.Executor(place)
        exe.run(fluid.default_startup_program())

        g_train_prog = fluid.CompiledProgram(g_train.program)
        d_train_prog = fluid.CompiledProgram(d_train.program)

        train_history = defaultdict(list)
        test_history = defaultdict(list)

        for epoch in range(1, self.cfg.epochs + 1):
            print('Epoch {}/{}'.format(epoch, self.cfg.epochs))

            num_batches = int(np.ceil(60000 / float(self.cfg.batch_size)))
            progress_bar = Bar('Training', max=num_batches)

            epoch_gen_loss = []
            epoch_disc_loss = []

            train_reader = paddle.batch(paddle.reader.shuffle(mnist.train(), buf_size=60000), 
                batch_size=self.cfg.batch_size, drop_last=True)
            test_reader = mnist.test()

            step = 0
            for i, data in enumerate(train_reader()):
              
                image_batch = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                label_batch = np.array([[x[1]] for x in data]).astype('int64')

                if len(image_batch) != self.cfg.batch_size:
                    continue

                # generate a new batch of noise
                noise_np = np.random.uniform(
                    -1, 1, (self.cfg.batch_size, self.cfg.latent_size)).astype('float32')

                # sample some labels from p_c
                sampled_labels_np = np.random.randint(
                    0, self.cfg.num_classes, self.cfg.batch_size).astype('int64')
                sampled_labels_np = np.expand_dims(sampled_labels_np, axis=1)

                # generate a batch of fake images, using the generated labels as
                # a conditioner. We reshape the sampled labels to be
                # (self.cfg.batch_size, 1) so that we can feed them into the 
                # embedding layer as a length one sequence

                generated_images = exe.run(g_train.infer_program,
                    feed={'sampled_labels' : sampled_labels_np, 'noise' : noise_np},
                    fetch_list=[g_train.fake_img])[0]

                x_np = np.concatenate((image_batch, generated_images))

                # use one-sided soft real/fake labels
                # Salimans et al., 2016
                # https://arxiv.org/pdf/1606.03498.pdf (Section 3.4)
                soft_zero, soft_one = 0, 0.95
                y_np = np.array([[soft_one]] * len(image_batch) + [[soft_zero]] * len(image_batch)).astype('float32')
                aux_y_np = np.concatenate((label_batch, sampled_labels_np), axis=0)

                # see if the discriminator can figure itself out...
                epoch_disc_loss.append(exe.run(d_train_prog, 
                    feed={'x' : x_np, 'y' : y_np, 'aux_y' : aux_y_np},
                    fetch_list=[d_train.loss])[0])

                # make new noise. we generate 2 * batch size here such that we have
                # the generator optimize over an identical number of images as the
                # discriminator

                noise_np = np.random.uniform(
                    -1, 1, (2 * self.cfg.batch_size, self.cfg.latent_size)).astype('float32')
                sampled_labels_np = np.random.randint(
                    0, self.cfg.num_classes, 2 * self.cfg.batch_size).astype('int64')
                sampled_labels_np = np.expand_dims(sampled_labels_np, axis=1)

                # we want to train the generator to trick the discriminator
                # For the generator, we want all the {fake, not-fake} labels to say
                # not-fake
                trick_np = np.array([[soft_one]] * 2 * self.cfg.batch_size).astype('float32')

                epoch_gen_loss.append(exe.run(g_train_prog,
                    feed={'sampled_labels' : sampled_labels_np, 
                          'noise' : noise_np, 
                          'trick' : trick_np},
                    fetch_list=[g_train.loss])[0])

                step += 1
                progress_bar.next()
            progress_bar.finish()

            print('Testing for epoch {}'.format(epoch))

            # evaluate the testing loss here

            # generate a new batch of noise
            noise_np = np.random.uniform(
                -1, 1, (self.cfg.test_size, self.cfg.latent_size)).astype('float32')

            # sample some labels from p_c and generate images from them
            sampled_labels_np = np.random.randint(
                0, self.cfg.num_classes, self.cfg.test_size).astype('int64')
            sampled_labels_np = np.expand_dims(sampled_labels_np, axis=1)

            generated_images = exe.run(g_train.infer_program,
                feed={'sampled_labels' : sampled_labels_np, 'noise' : noise_np},
                fetch_list=[g_train.fake_img])[0]
            
            x_test, y_test = [], []
            for data in test_reader():
                x_test.append(np.reshape(data[0], [1, 28, 28]))
                y_test.append([data[1]])
                if len(x_test) >= self.cfg.test_size:
                    break
            x_test = np.array(x_test).astype('float32')
            y_test = np.array(y_test).astype('int64')

            x_np = np.concatenate((x_test, generated_images))
            y_np = np.array([[1]] * self.cfg.test_size + [[0]] * self.cfg.test_size).astype('float32')
            aux_y_np = np.concatenate((y_test, sampled_labels_np), axis=0)

            # see if the discriminator can figure itself out...
            discriminator_test_loss = exe.run(d_train.infer_program, 
                feed={'x' : x_np, 'y' : y_np, 'aux_y' : aux_y_np},
                fetch_list=[d_train.unweighted_loss])[0][0]

            discriminator_train_loss = np.mean(np.array(epoch_disc_loss))

            # make new noise
            noise_np = np.random.uniform(
                -1, 1, (2 * self.cfg.test_size, self.cfg.latent_size)).astype('float32')
            sampled_labels_np = np.random.randint(
                0, self.cfg.num_classes, 2 * self.cfg.test_size).astype('int64')
            sampled_labels_np = np.expand_dims(sampled_labels_np, axis=1)

            trick_np = np.array([[1]] * 2 * self.cfg.test_size).astype('float32')

            generated_images = exe.run(g_train.infer_program,
                feed={'sampled_labels' : sampled_labels_np, 'noise' : noise_np},
                fetch_list=[g_train.fake_img])[0]
            generator_test_loss = exe.run(d_train.infer_program, 
                feed={'x' : generated_images, 'y' : trick_np, 'aux_y' : sampled_labels_np},
                fetch_list=[d_train.unweighted_loss])[0][0]

            generator_train_loss = np.mean(np.array(epoch_gen_loss))

            # generate an epoch report on performance
            train_history['generator'].append(generator_train_loss)
            train_history['discriminator'].append(discriminator_train_loss)

            test_history['generator'].append(generator_test_loss)
            test_history['discriminator'].append(discriminator_test_loss)

            print('train g loss', generator_train_loss)
            print('train d loss', discriminator_train_loss)
            print('test g loss', generator_test_loss)
            print('test d loss', discriminator_test_loss)

            # generate some digits to display
            num_rows = 4
            noise_np = np.tile(np.random.uniform(-1, 1, (num_rows, self.cfg.latent_size)),
                            (self.cfg.num_classes, 1)).astype('float32')

            sampled_labels_np = np.array(
                [[i] * num_rows for i in range(self.cfg.num_classes)]).reshape(-1, 1).astype('int64')

            generated_images = exe.run(g_train.infer_program,
                feed={'sampled_labels' : sampled_labels_np, 'noise' : noise_np},
                fetch_list=[g_train.fake_img])[0]

            def save_images(generated_images, epoch):
                for i in range(len(generated_images)):
                    fname = './data/image_epoch_%d_%d.jpeg' % (epoch, i)
                    img = np.array(generated_images[i]).astype('float32').reshape((28, 28))
                    img = img * 127.5 + 127.5
                    img = np.clip(img, 0, 255).astype('uint8')
                    img = Image.fromarray(img, 'L')
                    img.save(fname, format='JPEG')

            save_images(generated_images, epoch)

        with open('acgan-history.pkl', 'wb') as f:
            pickle.dump({'train': train_history, 'test': test_history}, f)

class Config(object):
    __slots__ = ('epochs',
                 'num_classes',
                 'batch_size', 
                 'latent_size', 
                 'adam_lr', 
                 'adam_beta_1',
                 'test_size')
                
if __name__ == '__main__':

    cfg = Config()

    # batch and latent size taken from the paper
    cfg.epochs = 10
    cfg.batch_size = 100
    cfg.latent_size = 100

    # Adam parameters suggested in https://arxiv.org/abs/1511.06434
    cfg.adam_lr = 0.0002
    cfg.adam_beta_1 = 0.5

    cfg.num_classes = 10
    cfg.test_size = 50

    model = MnistACGAN(cfg)
    model.build_model()

