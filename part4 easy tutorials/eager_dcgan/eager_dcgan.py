from __future__ import print_function
import random
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC, BatchNorm
from paddle.fluid.dygraph.base import to_variable
from layers import conv2d, DeConv2D
from PIL import Image
from visualdl import LogWriter

buffer_size = 60000
batch_size = 256
num_epochs = 150
noise_dim = 100
num_examples_to_generate = 4

class generator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(generator, self).__init__(name_scope)

        self.fc1 = FC(self.full_name(), 
                      size=7 * 7 * 64,
                      param_attr=fluid.initializer.Xavier(),
                      bias_attr=False)

        self.bn1 = BatchNorm(self.full_name(),
                             momentum=0.99,
                             num_channels=7 * 7 * 64,
                             param_attr=fluid.initializer.Xavier(),
                             bias_attr=fluid.initializer.Xavier(),
                             trainable_statistics=True)

        self.deconv1 = DeConv2D(self.full_name(),
                                num_filters=64,
                                filter_size=5,
                                stride=1,
                                padding=[2, 2],
                                relu=True,
                                norm=True,
                                use_bias=False)

        self.deconv2 = DeConv2D(self.full_name(),
                                num_filters=32,
                                filter_size=5,
                                stride=2,
                                padding=[1, 1],
                                relu=True,
                                norm=True,
                                use_bias=False)

        self.deconv3 = DeConv2D(self.full_name(),
                                num_filters=1,
                                filter_size=5,
                                stride=2,
                                padding=[2, 2],
                                relu=False,
                                norm=False,
                                use_bias=False)

    def forward(self, inputs, testing=False):
        x = self.fc1(inputs)
        x = self.bn1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.3)
        x = fluid.layers.reshape(x, shape=[-1, 64, 7, 7])
        x = self.deconv1(x)
        x = self.deconv2(x)
        x = self.deconv3(x)
        x = fluid.layers.crop(x, shape=(-1, 1, 28, 28))
        x = fluid.layers.tanh(x)
        return x

class discriminator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(discriminator, self).__init__(name_scope)

        # conv2d + leaky_relu + dropout
        self.conv1 = conv2d(self.full_name(),
                            num_filters=64,
                            filter_size=5,
                            stride=2,
                            padding=2,
                            relu=True,
                            dropout=0.3)

        # conv2d + leaky_relu 
        self.conv2 = conv2d(self.full_name(),
                            num_filters=128,
                            filter_size=5,
                            stride=2,
                            relu=True,
                            padding=2)

        self.fc1 = FC(self.full_name(), 
                      size=2,
                      param_attr=fluid.initializer.Xavier(),
                      bias_attr=fluid.initializer.Xavier())

    def forward(self, inputs):
        x = self.conv1(inputs)
        x = self.conv2(x)
        x = self.fc1(x)
        return x

def discriminator_loss(real_output, generated_output):
    real_ones_like = fluid.layers.ones(shape=[batch_size, 1], dtype='int64')
    generated_zeros_like = fluid.layers.zeros(shape=[batch_size, 1], dtype='int64')
    real_loss = fluid.layers.softmax_with_cross_entropy(real_output, real_ones_like)
    generated_loss = fluid.layers.softmax_with_cross_entropy(generated_output, generated_zeros_like)
    return fluid.layers.elementwise_add(real_loss, generated_loss)

def generator_loss(generated_output):
    generated_ones_like = fluid.layers.ones(shape=[batch_size, 1], dtype='int64')
    generated_loss = fluid.layers.softmax_with_cross_entropy(generated_output, generated_ones_like)
    return generated_loss

class dcgan(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(dcgan, self).__init__(name_scope)

        self.build_generator = generator(self.full_name())
        self.build_discriminator = discriminator(self.full_name())

    def forward(self, noise, real=None, generated=None, is_gen=True, is_train=True):
        if is_gen:
            generated_images = self.build_generator(noise)
            if is_train:
                disc_generated_output = self.build_discriminator(generated_images)
                gen_loss = generator_loss(disc_generated_output)
                return gen_loss, generated_images
            else:
                return generated_images
        else:
            disc_real_output = self.build_discriminator(real)
            disc_generated_output = self.build_discriminator(generated)
            disc_loss = discriminator_loss(disc_real_output, disc_generated_output)
            return disc_loss


def generate_and_save_images(epoch, model, random_vector_for_generation):
    predictions = model(random_vector_for_generation, None, None, True, False)

    for i in range(num_examples_to_generate):
        fname = './data/image_epoch_%d_%d.jpeg' % (epoch, i)
        img = np.array(predictions.numpy()[i]).astype('float32').reshape((28, 28))
        img = img * 127.5 + 127.5
        img = np.clip(img, 0, 255).astype('uint8')
        img = Image.fromarray(img, 'L')
        img.save(fname, format='JPEG')

def train():
    log_writter = LogWriter('./vdl_log', sync_cycle=10) 

    with log_writter.mode("train") as logger:          
        log_g_loss = logger.scalar(tag="g_loss") 
        log_d_loss = logger.scalar(tag="d_loss")

    place = fluid.CUDAPlace(1)
    with fluid.dygraph.guard(place):

        random_vector_data = np.random.standard_normal((num_examples_to_generate, noise_dim)).astype('float32')
        random_vector_for_generation = to_variable(random_vector_data)

        mnist_dcgan = dcgan('mnist_dcgan')

        discriminator_optimizer = fluid.optimizer.Adam(learning_rate=1e-4)
        generator_optimizer = fluid.optimizer.Adam(learning_rate=1e-4)

        train_data = paddle.dataset.mnist.train()

        for epoch in range(num_epochs):

            train_reader = paddle.batch(paddle.reader.shuffle(train_data, buf_size=buffer_size), 
                batch_size=batch_size, drop_last=True)

            print("Epoch id: ", epoch)

            total_loss_gen = []
            total_loss_disc = []

            for batch_id, data in enumerate(train_reader()):

                noise_data = np.random.standard_normal((batch_size, noise_dim)).astype('float32')
                noise = to_variable(noise_data)

                img_data = np.array([x[0].reshape(1, 28, 28) for x in data]).astype('float32')
                img = to_variable(img_data)

                gen_loss, generated_images = mnist_dcgan(noise, img, None, True)
                gen_loss = fluid.layers.reduce_mean(gen_loss)

                gen_loss.backward()
                vars_G = []
                for parm in mnist_dcgan.parameters():
                    if parm.name[:31] == 'mnist_dcgan/dcgan_0/generator_0':
                        vars_G.append(parm)
                generator_optimizer.minimize(gen_loss, parameter_list=vars_G)
                mnist_dcgan.clear_gradients()



                disc_loss = mnist_dcgan(noise, img, generated_images, False)
                disc_loss = fluid.layers.reduce_mean(disc_loss)

                disc_loss.backward()
                vars_D = []
                for parm in mnist_dcgan.parameters():
                    if parm.name[:35] == 'mnist_dcgan/dcgan_0/discriminator_0':
                        vars_D.append(parm)
                discriminator_optimizer.minimize(disc_loss, parameter_list=vars_D)
                mnist_dcgan.clear_gradients()

                total_loss_gen.append(gen_loss.numpy()[0])
                total_loss_disc.append(disc_loss.numpy()[0])

            if epoch % 10 == 0:
                generate_and_save_images(epoch, mnist_dcgan, random_vector_for_generation)

            print("Generator loss: ", 
                np.mean(np.array(total_loss_gen).astype('float32')))
            print("Discriminator loss: ", 
                np.mean(np.array(total_loss_disc).astype('float32')))

            log_g_loss.add_record(epoch, np.mean(np.array(total_loss_gen).astype('float32')))
            log_d_loss.add_record(epoch, np.mean(np.array(total_loss_disc).astype('float32')))

if __name__ == '__main__':
    train()
            
