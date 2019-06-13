#encoding=utf8

import os
import sys
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid

from arg_config import ArgConfig, print_arguments
from cnn_mnist_net import create_net

def init_from_params(args, exe, program):
    
    assert isinstance(args.init_from_params, str)

    if not os.path.exists(args.init_from_params):
        raise Warning("the checkpotin path does not exist.")
        return False

    fluid.io.load_persistables(executor = exe, dirname=args.init_from_params, main_program = program)
    print("init model from params at %s" % (args.init_from_params))

    return True

def do_save_inference_model(args):

    test_prog = fluid.default_main_program()
    startup_prog = fluid.default_startup_program()

    with fluid.program_guard(test_prog, startup_prog):
        test_prog.random_seed = args.random_seed
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
                paddle.dataset.mnist.test(),
                batch_size = args.batch_size)

            reader.decorate_sample_list_generator(generator)
            
            # define the network

            prediction = create_net(is_training = False, model_input = image, args = args)

            prediction = fluid.layers.argmax(prediction, axis=-1)


    # prepare predicting

    if args.use_cuda:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()

    exe = fluid.Executor(place)

    exe.run(startup_prog)

    if args.init_from_params:
        init_from_params(args, exe, test_prog)
    else:
        raise ValueError("The prediction model shall be inited before prediction, please use ``--init_from_params`` to identity the model path.")

    # save inference model for depolyment
    fluid.io.save_inference_model(
        args.inference_model_dir,
        feeded_var_names = [image.name],
        target_vars = [prediction],
        executor = exe,
        main_program = test_prog)

    print("save inference model at %s" % (args.inference_model_dir))

if __name__ == "__main__":
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    do_save_inference_model(args)

