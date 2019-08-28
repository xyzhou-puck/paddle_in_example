from __future__ import print_function

import os
import argparse
from PIL import Image
import numpy
import paddle
import paddle.fluid as fluid
from SimpleRNN import simple_rnn

# Training parameters.
BATCH_SIZE = 32
NUM_CLASSES = 10
EPOCHS = 200
HIDDEN_UNITS = 100

learning_rate = 1e-6
clip_norm = 1.0

def net_conf(img, label, num_classes):

    img = fluid.layers.reshape(img, [-1, 28, 28])

    _, last_hidden = simple_rnn(rnn_input=img,
                                init_hidden=None,
                                hidden_size=HIDDEN_UNITS,
                                kernel_param_attr=fluid.initializer.Normal(scale=0.001),
                                recurrent_param_attr=None,
                                act='relu')

    prediction = fluid.layers.fc(input=last_hidden, size=NUM_CLASSES, act='softmax')

    loss = fluid.layers.cross_entropy(input=prediction, label=label)
    avg_loss = fluid.layers.mean(loss)
    acc = fluid.layers.accuracy(input=prediction, label=label)
    return prediction, avg_loss, acc

def train(train_dataset, test_dataset):
    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()
    inference_program = fluid.default_main_program().clone(for_test=True)

    train_reader = paddle.batch(paddle.reader.shuffle(train_dataset, buf_size=500), batch_size=BATCH_SIZE)
    test_reader = paddle.batch(paddle.reader.shuffle(test_dataset, buf_size=500), batch_size=BATCH_SIZE)

    img = fluid.layers.data(name='img', shape=[1, 28, 28], dtype='float32')
    label = fluid.layers.data(name='label', shape=[1], dtype='int64')

    prediction, avg_loss, acc = net_conf(img, label, NUM_CLASSES)

    optimizer = fluid.optimizer.RMSProp(learning_rate=learning_rate)
    optimizer.minimize(avg_loss)

    place = fluid.CPUPlace()
    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[img, label], place=place)
    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(EPOCHS)]

    # train
    step = 0
    for epoch_id in epochs:
        print("Epoch %d" % (epoch_id))
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(main_program, feed=feeder.feed(data), fetch_list=[avg_loss, acc])
            if step % 100 == 0:
                print("Pass %d, Cost %f, Acc %f" % (step, metrics[0], metrics[1]))
            step += 1

    # test
    total_acc = 0
    step = 0
    for step_id, data in enumerate(test_reader()):
        metrics = exe.run(inference_program, feed=feeder.feed(data), fetch_list=[avg_loss, acc])
        total_acc += metrics[1]
        step += 1

    print("Acc: %f" % (total_acc / step))


def main():
    mnist_dataset_train = paddle.dataset.mnist.train()
    mnist_dataset_test = paddle.dataset.mnist.test()
    train(mnist_dataset_train, mnist_dataset_test)

if __name__ == '__main__':
    main()
