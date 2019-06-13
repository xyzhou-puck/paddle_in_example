#encoding=utf8

import os
import sys
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid

from arg_config import ArgConfig, print_arguments
from cnn_mnist_net import create_net

def init_from_checkpoint(args, exe, program):
    
    assert isinstance(args.init_from_checkpoint, str)

    if not os.path.exists(args.init_from_checkpoint):
        raise Warning("the checkpotin path does not exist.")
        return False

    fluid.io.load_persistables(executor = exe, dirname=args.init_from_checkpoint, main_program = program)
    print("init model from checkpoint at %s" % (args.init_from_checkpoint))

    return True

def save_checkpoint(args, exe, program, dirname):
    
    assert isinstance(args.save_model_path, str)

    checkpoint_dir = args.save_model_path + "/" + args.save_checkpoint

    if not os.path.exists(checkpoint_dir):
        os.mkdir(checkpoint_dir)

    fluid.io.save_persistables(exe, checkpoint_dir + "/" + dirname, main_program = program)
    print("save checkpoint at %s" % (checkpoint_dir + "/" + dirname))

    return True

def save_param(args, exe, program, dirname):

    assert isinstance(args.save_model_path, str)
    
    param_dir = args.save_model_path + "/" + args.save_param

    if not os.path.exists(param_dir):
        os.mkdir(param_dir)

    fluid.io.save_params(exe, param_dir + "/" + dirname, main_program = program)
    print("save parameters at %s" % (param_dir + "/" + dirname))

    return True


def do_train(args):

    train_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(train_prog, startup_prog):
        train_prog.random_seed = args.random_seed
        startup_prog.random_seed = args.random_seed

        with fluid.unique_name.guard():
            
            # define reader

            image = fluid.layers.data(
                name='image', shape=[1, 28, 28], dtype='float32')

            label = fluid.layers.data(
                name='label', shape=[1], dtype='int64')

            reader = fluid.io.PyReader(feed_list=[image, label],
                capacity=4, iterable=False)

            generator = paddle.batch(
                paddle.reader.shuffle(
                    paddle.dataset.mnist.train(),
                    buf_size = 500),
                batch_size = args.batch_size)

            reader.decorate_sample_list_generator(generator)

            
            # define the network

            loss, prediction = create_net(is_training = True, model_input = [image, label], args = args)

            # define optimizer for learning

            optimizer = fluid.optimizer.Adam(learning_rate=args.learning_rate)
            optimizer.minimize(loss)


    # prepare training

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    exe.run(startup_prog)

    if args.init_from_checkpoint:
        init_from_checkpoint(args, exe, train_prog)

    compiled_train_prog = fluid.CompiledProgram(train_prog).with_data_parallel(
        loss_name = loss.name)

    # start training

    step = 0
    for epoch_step in range(args.epoch_num):
        reader.start()
        while True:
            try:

                # this is for minimizing the fetching op, saving the training speed.
                if step % args.print_step == 0:
                    fetch_list = [loss.name]
                else:
                    fetch_list = []

                output = exe.run(compiled_train_prog, fetch_list = fetch_list)

                if step % args.print_step == 0:
                    print("step: %d, loss: %.4f" % (step, np.sum(output[0])))

                if step % args.save_step == 0 and step != 0:

                    if args.save_checkpoint:
                        save_checkpoint(args, exe, train_prog, "step_" + str(step))

                    if args.save_param:
                        save_param(args, exe, train_prog, "step_" + str(step))

                step += 1

            except fluid.core.EOFException:
                reader.reset()
                break

    if args.save_checkpoint:
        save_checkpoint(args, exe, train_prog, "step_final")

    if args.save_param:
        save_param(args, exe, train_prog, "step_final")



if __name__ == "__main__":
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    do_train(args)

