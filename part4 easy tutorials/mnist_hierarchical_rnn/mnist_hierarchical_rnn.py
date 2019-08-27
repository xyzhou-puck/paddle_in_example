from __future__ import print_function
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.layers import basic_lstm

import numpy as np
import argparse

# Training parameters.
BATCH_SIZE = 32
num_classes = 10
epoch = 5

# Embedding dimensions.
row_hidden = 128
col_hidden = 128

def parse_args():
    parser = argparse.ArgumentParser("mnist")
    parser.add_argument(
        '--enable_ce',
        action='store_true',
        help="If set, run the task with continuous evaluation logs.")
    parser.add_argument(
        '--use_gpu',
        type=bool,
        default=True,
        help="Whether to use GPU or not.")
    args = parser.parse_args()
    return args


def hierarchical_rnn_neural_network(img, label):

    img = (img + 1) / 2 # [-1, 1] --> [0, 1]
    encoded_rows, _, _ = basic_lstm(img, None, None, row_hidden)

    _, encoded_columns, _ = basic_lstm(encoded_rows, None, None, col_hidden)
    prediction = fluid.layers.fc(encoded_columns[0], num_classes, act='softmax')
    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    accuracy = fluid.layers.accuracy(input=prediction, label=label)

    return avg_loss, accuracy

def train(args):
    if args.use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    enable_ce = False
    if enable_ce:
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

    img = fluid.layers.data(name='img', shape=[28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    net_conf = hierarchical_rnn_neural_network

    loss, accuracy = net_conf(img, label)

    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.RMSProp(learning_rate=0.001)
    optimizer.minimize(loss)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        loss_set = []
        acc_set = []
        for test_data in train_test_reader():
            val_loss_np, val_acc_np = exe.run(
                program=train_test_program,
                feed=train_test_feed.feed(test_data),
                fetch_list=[loss, accuracy])
            loss_set.append(float(val_loss_np))
            acc_set.append(float(val_acc_np))
        # get test acc and loss
        loss_mean = np.array(loss_set).mean()
        acc_mean = np.array(acc_set).mean()

        return loss_mean, acc_mean

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(epoch)]

    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(
                main_program,
                feed=feeder.feed(data),
                fetch_list=[loss, accuracy])
            if step % 200 == 0:
                print("Pass %d, Epoch %d, loss %f, acc %f" % (step, epoch_id,
                                                      metrics[0], metrics[1]))
            step += 1
        # test for epoch
        val_loss, val_acc = train_test(
            train_test_program=test_program,
            train_test_reader=test_reader,
            train_test_feed=feeder)

        print("Test with Epoch %d, val_loss: %s, val_acc: %s" %
              (epoch_id, val_loss, val_acc))
        lists.append((epoch_id, loss, val_acc))

    # find the best pass
    best = sorted(lists, key=lambda list: float(list[2]))[-1]
    print('Best pass is %s,  val_acc: %s' % (best[0],best[2]))

def main():

    parser = argparse.ArgumentParser()
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
