from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
import argparse


# reparameterization trick
# instead of sampling from Q(z|X), sample eps = N(0,I)
# z = z_mean + sqrt(var)*eps
def sampling(z_mean, z_log_var):
    """Reparameterization trick by sampling fr an isotropic unit Gaussian.

    # Arguments
        args (tensor): mean and log of variance of Q(z|X)

    # Returns
        z (tensor): sampled latent vector
    """

    # by default, random_normal has mean=0 and std=1.0
    epsilon = fluid.layers.gaussian_random_batch_size_like(z_mean, shape=[-1, latent_dim])
    epsilon.stop_gradient = True
    return z_mean + fluid.layers.exp(0.5 * z_log_var) * epsilon


intermediate_dim = 512
BATCH_SIZE = 128
latent_dim = 2
epochs = 50

def vae_neural_network(img, args):
    original_dim = img.shape[-1] * img.shape[-2]
    img = fluid.layers.reshape(img, shape=[-1, original_dim])
    img = (img + 1) / 2 # [-1, 1] --> [0, 1]
    x = fluid.layers.fc(input=img, size=intermediate_dim, act='relu')
    z_mean = fluid.layers.fc(input=x, size=latent_dim)
    z_log_var = fluid.layers.fc(input=x, size=latent_dim)

    z = sampling(z_mean, z_log_var)

    x = fluid.layers.fc(input=z, size=intermediate_dim, act='relu')
    outputs = fluid.layers.fc(input=x, size=original_dim, act='sigmoid')

    if args.mse:
        reconstruction_loss = fluid.layers.square_error_cost(input=outputs, label=img)
    else:
        outputs = fluid.layers.clip(outputs, 0.01, 0.99)
        outputs = fluid.layers.log(outputs / (1 - outputs))
        reconstruction_loss = fluid.layers.sigmoid_cross_entropy_with_logits(x=outputs, label=img)
    reconstruction_loss = fluid.layers.reduce_mean(reconstruction_loss, dim=-1)

    reconstruction_loss *= original_dim

    kl_loss = 1 + z_log_var - fluid.layers.square(z_mean) - fluid.layers.exp(z_log_var)
    kl_loss = fluid.layers.reduce_sum(kl_loss, dim=-1)
    kl_loss *= -0.5
    vae_loss = reconstruction_loss + kl_loss
    reconstruction_loss = fluid.layers.reduce_mean(reconstruction_loss)
    kl_loss = fluid.layers.reduce_mean(kl_loss)
    vae_loss = fluid.layers.reduce_mean(vae_loss)
    return outputs, reconstruction_loss, kl_loss, vae_loss

def train(args):

    if args.use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    if args.enable_ce:
        train_reader = paddle.batch(
            paddle.dataset.mnist.train(), batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)
        startup_program.random_seed = 90
        main_program.random_seed = 90
    else:
        train_reader = paddle.batch(
            paddle.reader.shuffle(paddle.dataset.mnist.train(), buf_size=500),
            batch_size=BATCH_SIZE)
        test_reader = paddle.batch(
            paddle.dataset.mnist.test(), batch_size=BATCH_SIZE)

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net_conf = vae_neural_network

    _, reconstruction_loss, kl_loss, vae_loss = net_conf(img, args)

    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(vae_loss)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        reconstruction_loss_set = []
        kl_loss_set = []
        vae_loss_set = []
        for test_data in train_test_reader():
            reconstruction_loss_np, kl_loss_np, vae_loss_np = exe.run(
                program=train_test_program,
                feed=train_test_feed.feed(test_data),
                fetch_list=[reconstruction_loss, kl_loss, vae_loss])
            reconstruction_loss_set.append(float(reconstruction_loss_np))
            kl_loss_set.append(float(kl_loss_np))
            vae_loss_set.append(float(vae_loss_np))
        # get test acc and loss
        reconstruction_loss_mean = np.array(reconstruction_loss_set).mean()
        kl_loss_mean = np.array(kl_loss_set).mean()
        vae_loss_mean = np.array(vae_loss_set).mean()
        return reconstruction_loss_mean, kl_loss_mean, vae_loss_mean

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(50)]

    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(
                main_program,
                feed=feeder.feed(data),
                fetch_list=[reconstruction_loss, kl_loss, vae_loss])
            if step % 100 == 0:
                print("Pass %d, Epoch %d, reconstruction_loss %f, kl_loss %f, vae_loss %f" % (step, epoch_id,
                                                      metrics[0], metrics[1], metrics[2]))
            step += 1
        # test for epoch
        reconstruction_loss_val, kl_loss_val, vae_loss_val = train_test(
            train_test_program=test_program,
            train_test_reader=test_reader,
            train_test_feed=feeder)

        print("Test with Epoch %d, reconstruction_loss_val: %s, kl_loss_val: %s, vae_loss_val: %s" %
              (epoch_id, reconstruction_loss_val, kl_loss_val, vae_loss_val))
        lists.append((epoch_id, reconstruction_loss_val, kl_loss_val, vae_loss_val))

    # find the best pass
    # best = sorted(lists, key=lambda list: float(list[3]))[0]
    # print('Best pass is %s, vae_loss_val is %s' % (best[0], best[3]))

def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("-m", "--mse",
                        help="Use mse loss instead of binary cross entropy (default)",
                        action='store_true')
    parser.add_argument("-c", "--enable_ce",
                        help="Whether to enable ce",
                        action='store_true')
    parser.add_argument("-g", "--use_cuda",
                        help="Whether to use GPU to train",
                        default=True)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
