from __future__ import print_function
import numpy as np
import paddle
import paddle.fluid as fluid
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Conv2DTranspose, Pool2D, FC, BatchNorm
from paddle.fluid.dygraph.base import to_variable
from PIL import Image
import cv2
import os

use_cudnn = False
restore = True
data_dir = './facades'
buffer_size = 400
batch_size = 1
img_width = 256
img_height = 256
num_epochs = 200

def random_crop(img_a, img_b):
    height, width = img_a.shape[:2]
    dh = int((height - img_height) * np.random.uniform())
    dw = int((width - img_width) * np.random.uniform())
    img_a_crop = img_a[dh : dh + img_height, dw : dw + img_width, :]
    img_b_crop = img_b[dh : dh + img_height, dw : dw + img_width, :]
    return img_a_crop, img_b_crop

def load_image(image_file, is_train=True):
    img = cv2.imread(image_file)
    height, width = img.shape[:2]
    width_2 = int(width / 2)
    real_img = img[0 : height, 0 : width_2, :]
    input_img = img[0 : height, width_2 : width, :]

    if is_train:
        real_img = cv2.resize(real_img, (286, 286))
        input_img = cv2.resize(input_img, (286, 286))
        real_img, input_img = random_crop(real_img, input_img)
        if np.random.uniform() > 0.5:
            real_img = cv2.flip(real_img, 1)
            input_img = cv2.flip(input_img, 1)
    else:
        real_img = cv2.resize(real_img, (img_height, img_width))
        input_img = cv2.resize(input_img, (img_height, img_width))

    real_img = np.array(real_img).astype('float32').transpose((2, 0, 1))
    input_img = np.array(input_img).astype('float32').transpose((2, 0, 1))

    real_img = (real_img / 127.5) - 1.0
    input_img = (input_img / 127.5) - 1.0

    return real_img, input_img

def prepare_dataset(directory, is_train=True):
    real_imgs = []
    input_imgs = []

    print('prepare data...')

    if is_train:
        directory += '/train'
    else:
        directory += '/test'
    for filename in os.listdir(directory):
        if filename.endswith(".jpg"): 
            fname = os.path.join(directory, filename)
            real_img, input_img = load_image(fname, is_train=is_train)
            real_img = np.expand_dims(real_img, axis=0)
            input_img = np.expand_dims(input_img, axis=0)
            real_imgs.append(real_img)
            input_imgs.append(input_img)

    real_imgs = np.concatenate(real_imgs)
    input_imgs = np.concatenate(input_imgs)

    print('load data size: ', len(real_imgs))

    return real_imgs, input_imgs

def batch_generator(real_imgs, input_imgs, batch_size, drop_last=True):
    batch_real_imgs, batch_input_imgs = [], []
    data = zip(real_imgs, input_imgs)
    for real_img, input_img in data:
        batch_real_imgs.append(real_img)
        batch_input_imgs.append(input_img)

        if len(batch_real_imgs) >= batch_size:
            yield np.array(batch_real_imgs).astype("float32"), \
                np.array(batch_input_imgs).astype("float32")
            batch_real_imgs, batch_input_imgs = [], []
    if batch_real_imgs and not drop_last:
        yield np.array(batch_real_imgs).astype("float32"), \
            np.array(batch_input_imgs).astype("float32")

class downsample(fluid.dygraph.Layer):
    def __init__(self, name_scope, filters, size, apply_batchnorm=True):
        super(downsample, self).__init__(name_scope)

        self.conv1 = Conv2D(
            self.full_name(), 
            num_filters=filters,
            filter_size=size,
            stride=2,
            padding=1,
            use_cudnn=use_cudnn,
            param_attr=fluid.initializer.Normal(0.0, 0.2),
            bias_attr=False)

        if apply_batchnorm:
            self.bn1 = BatchNorm(
                self.full_name(), 
                num_channels=filters)

        self.filters = filters
        self.size = size
        self.apply_batchnorm = apply_batchnorm

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.apply_batchnorm:
            x = self.bn1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.3)
        return x

class upsample(fluid.dygraph.Layer):
    def __init__(self, name_scope, filters, size, apply_dropout=False):
        super(upsample, self).__init__(name_scope)

        self.deconv1 = Conv2DTranspose(
            self.full_name(),
            num_filters=filters,
            filter_size=size,
            stride=2,
            padding=1,
            use_cudnn=use_cudnn,
            param_attr=fluid.initializer.Normal(),
            bias_attr=False)

        self.bn1 = BatchNorm(
            self.full_name(), 
            num_channels=filters)

        self.filters = filters
        self.size = size
        self.apply_dropout = apply_dropout

    def forward(self, inputs):
        x1, x2 = inputs
        x = self.deconv1(x1)
        x = self.bn1(x)
        if self.apply_dropout:
            x = fluid.layers.dropout(x, dropout_prob=0.5)
        x = fluid.layers.relu(x)
        concat = fluid.layers.concat([x, x2], axis=1)
        return concat

class generator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(generator, self).__init__(name_scope)

        self.down1 = downsample(self.full_name(), 64, 4, False)
        self.down2 = downsample(self.full_name(), 128, 4)
        self.down3 = downsample(self.full_name(), 256, 4)
        self.down4 = downsample(self.full_name(), 512, 4)
        self.down5 = downsample(self.full_name(), 512, 4)
        self.down6 = downsample(self.full_name(), 512, 4)
        self.down7 = downsample(self.full_name(), 512, 4)
        self.down8 = downsample(self.full_name(), 512, 4)

        self.up1 = upsample(self.full_name(), 512, 4, True)
        self.up2 = upsample(self.full_name(), 512, 4, True)
        self.up3 = upsample(self.full_name(), 512, 4, True)
        self.up4 = upsample(self.full_name(), 512, 4)
        self.up5 = upsample(self.full_name(), 256, 4)
        self.up6 = upsample(self.full_name(), 128, 4)
        self.up7 = upsample(self.full_name(), 64, 4)
        self.last = Conv2DTranspose(
            self.full_name(),
            num_filters=3,
            filter_size=4,
            stride=2,
            padding=1,
            use_cudnn=use_cudnn,
            param_attr=fluid.initializer.Normal(0.0, 0.2),
            act='tanh')

    def forward(self, inputs):
        x1 = self.down1(inputs)
        x2 = self.down2(x1)
        x3 = self.down3(x2)
        x4 = self.down4(x3)
        x5 = self.down5(x4)
        x6 = self.down6(x5)
        x7 = self.down7(x6)
        x8 = self.down8(x7)

        x9 = self.up1((x8, x7))
        x10 = self.up2((x9, x6))
        x11 = self.up3((x10, x5))
        x12 = self.up4((x11, x4))
        x13 = self.up5((x12, x3))
        x14 = self.up6((x13, x2))
        x15 = self.up7((x14, x1))
        x16 = self.last(x15)

        # print('generator output: ', x16.shape)

        return x16

class disc_downsample(fluid.dygraph.Layer):
    def __init__(self, name_scope, filters, size, apply_batchnorm=True):
        super(disc_downsample, self).__init__(name_scope)

        self.conv1 = Conv2D(
            self.full_name(), 
            num_filters=filters,
            filter_size=size,
            stride=2,
            padding=1,
            use_cudnn=use_cudnn,
            param_attr=fluid.initializer.Normal(0.0, 0.2),
            bias_attr=False)

        if apply_batchnorm:
            self.bn1 = BatchNorm(
                self.full_name(), 
                num_channels=filters)

        self.filters = filters
        self.size = size
        self.apply_batchnorm = apply_batchnorm

    def forward(self, inputs):
        x = self.conv1(inputs)
        if self.apply_batchnorm:
            x = self.bn1(x)
        x = fluid.layers.leaky_relu(x, alpha=0.3)
        return x

class discriminator(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(discriminator, self).__init__(name_scope)

        self.down1 = disc_downsample(self.full_name(), 64, 4, False)
        self.down2 = disc_downsample(self.full_name(), 128, 4)
        self.down3 = disc_downsample(self.full_name(), 256, 4)

        self.conv = Conv2D(
            self.full_name(), 
            num_filters=512,
            filter_size=4,
            stride=1,
            use_cudnn=use_cudnn,
            param_attr=fluid.initializer.Normal(),
            bias_attr=False)

        self.bn = BatchNorm(
            self.full_name(), 
            num_channels=512)

        self.last = Conv2D(
            self.full_name(), 
            num_filters=2,
            filter_size=4,
            stride=1,
            use_cudnn=use_cudnn,
            param_attr=fluid.initializer.Normal())

    def forward(self, inputs):
        x, y = inputs
        x = fluid.layers.concat([x, y], axis=1)
        x = self.down1(x)
        x = self.down2(x)
        x = self.down3(x)
        x = fluid.layers.pad2d(x, [1,1,1,1])
        x = self.conv(x)
        x = self.bn(x)
        x = fluid.layers.leaky_relu(x, alpha=0.3)
        x = fluid.layers.pad2d(x, [1,1,1,1])
        x = self.last(x)

        # print('discriminator output: ', x.shape)

        return x

def discriminator_loss(real_output, generated_output):
    real_output = fluid.layers.reshape(real_output, shape=[-1, 2])
    generated_output = fluid.layers.reshape(generated_output, shape=[-1, 2])

    real_ones_like = fluid.layers.ones(shape=[batch_size * 900, 1], dtype='int64')
    generated_zeros_like = fluid.layers.zeros(shape=[batch_size * 900, 1], dtype='int64')

    real_loss = fluid.layers.softmax_with_cross_entropy(
        real_output, real_ones_like)
    generated_loss = fluid.layers.softmax_with_cross_entropy(
        generated_output, generated_zeros_like)

    real_loss = fluid.layers.reshape(real_loss, shape=[batch_size, -1])
    generated_loss = fluid.layers.reshape(generated_loss, shape=[batch_size, -1])

    real_loss = fluid.layers.reduce_mean(real_loss)
    generated_loss = fluid.layers.reduce_mean(generated_loss)

    return real_loss + generated_loss

def generator_loss(disc_judgment, generated_output, target):
    disc_judgment = fluid.layers.reshape(disc_judgment, shape=[-1, 2])
    disc_judgment_ones_like = fluid.layers.ones(shape=[batch_size * 900, 1], dtype='int64')
    gan_loss = fluid.layers.softmax_with_cross_entropy(disc_judgment, disc_judgment_ones_like)

    gan_loss = fluid.layers.reshape(gan_loss, shape=[batch_size, -1])

    gan_loss = fluid.layers.reduce_mean(gan_loss)

    l1_loss = fluid.layers.reduce_mean(fluid.layers.abs(target - generated_output))
    return gan_loss + (100 * l1_loss)

class build_pix2pix_gan(fluid.dygraph.Layer):
    def __init__(self, name_scope):
        super(build_pix2pix_gan, self).__init__(name_scope)

        self.generator = generator(self.full_name())
        self.discriminator = discriminator(self.full_name())

    def forward(self, inputs, target, disc_generated, gen=True, test=False):
        if test:
            generated_images = self.generator(inputs)
            return generated_images
        elif gen:
            generated_images = self.generator(inputs)
            disc_generated_output = self.discriminator((inputs, generated_images))
            gen_loss = generator_loss(disc_generated_output, generated_images, target)
            return gen_loss, disc_generated_output
        else:
            disc_real_output = self.discriminator((inputs, target))
            disc_loss = discriminator_loss(disc_real_output, disc_generated)
            return disc_loss
            
def generate_and_save_images(generator, input, id):
    predictions = generator(input, None, None, False, True)
    fname = 'pix2pix_%d.jpeg' % id

    img = np.array(predictions.numpy()).reshape((3, 256, 256)).astype('float32')
    img = img * 127.5 + 127.5
    img = np.clip(img, 0, 255).astype('uint8')
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img, 'RGB')
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    img.save(fname, format='JPEG')

def train():
    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):

        pix2pix_gan = build_pix2pix_gan('pix2pix_gan')

        discriminator_optimizer = AdamOptimizer(learning_rate=2e-4, beta1=0.5)
        generator_optimizer = AdamOptimizer(learning_rate=2e-4, beta1=0.5)

        real_dataset, input_dataset = prepare_dataset(data_dir, is_train=True)
        real_test, input_test = prepare_dataset(data_dir, is_train=False)

        epoch = 0

        if os.path.exists('./model'):
            print('load prev checkpoint...')
            model, _ = fluid.dygraph.load_persistables('./model')
            pix2pix_gan.load_dict(model)
            checkpoint = open("./checkpoint.txt", "r")
            epoch = int(checkpoint.read()) + 1
            checkpoint.close()

        while epoch < num_epochs:

            print("Epoch id: ", epoch)

            total_loss_gen = 0
            total_loss_disc = 0  

            seed = np.random.randint(1000)
            np.random.seed(seed)
            np.random.shuffle(real_dataset)
            np.random.seed(seed)
            np.random.shuffle(input_dataset)

            for tar, inpt in batch_generator(real_dataset, input_dataset, batch_size): 

                target = to_variable(tar)
                input_image = to_variable(inpt)

                gen_loss, disc_generated = pix2pix_gan(input_image, target, None, True)
                gen_loss.backward()
                vars_G = []
                for parm in pix2pix_gan.parameters():
                    if parm.name[:43] == 'pix2pix_gan/build_pix2pix_gan_0/generator_0':
                        vars_G.append(parm)
                generator_optimizer.minimize(gen_loss, parameter_list=vars_G)
                pix2pix_gan.clear_gradients()

                disc_loss = pix2pix_gan(input_image, target, disc_generated, False)
                disc_loss.backward()
                vars_D = []
                for parm in pix2pix_gan.parameters():
                    if parm.name[:47] == 'pix2pix_gan/build_pix2pix_gan_0/discriminator_0':
                        vars_D.append(parm)
                discriminator_optimizer.minimize(disc_loss, parameter_list=vars_D)
                pix2pix_gan.clear_gradients()

                total_loss_gen += gen_loss.numpy()[0]
                total_loss_disc += disc_loss.numpy()[0]

            print("Total generator loss: ", total_loss_gen)
            print("Total discriminator loss: ", total_loss_disc)

            if epoch % 10 == 0:
                # save checkpoint
                fluid.dygraph.save_persistables(pix2pix_gan.state_dict(), "./model")
                checkpoint = open("./checkpoint.txt", "w")
                checkpoint.write(str(epoch))
                checkpoint.close()

                input_image = to_variable(input_test)
                generate_and_save_images(pix2pix_gan, input_image, epoch)

            epoch += 1

if __name__ == '__main__':
    train()