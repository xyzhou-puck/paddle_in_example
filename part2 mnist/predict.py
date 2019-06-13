#encoding=utf8

import os
import sys
import numpy as np
import argparse
import paddle
import paddle.fluid as fluid

from arg_config import ArgConfig, print_arguments
from cnn_mnist_net import create_net

def save_results(args, preds):

    assert isinstance(args.prediciton_dir, str)

    if not os.path.exists(args.prediciton_dir):
        os.mkdir(args.prediciton_dir)

    example_id = 0
    fout = open(args.prediciton_dir + "/prediction.txt", "w")
    for example in preds:
        fout.write("%d\t%d\n" % (example_id, example))
        example_id += 1

    fout.close()

    return True

def init_from_params(args, exe, program):
    
    assert isinstance(args.init_from_params, str)

    if not os.path.exists(args.init_from_params):
        raise Warning("the checkpotin path does not exist.")
        return False

    fluid.io.load_params(executor = exe, dirname=args.init_from_params, main_program = program)
    print("init model from params at %s" % (args.init_from_params))

    return True

def do_predict(args):

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

    # start predicting

    outputs = []
    reader.start()
    while True:
        try:

            fetch_list = [prediction.name]
            output = exe.run(test_prog, fetch_list = fetch_list)
            outputs.append(output[0])

        except fluid.core.EOFException:
            reader.reset()
            break

    outputs = np.concatenate(outputs).astype("int32")

    if args.prediciton_dir:
        save_results(args, outputs)

    return outputs


if __name__ == "__main__":
    args = ArgConfig()
    args = args.build_conf()
    print_arguments(args)

    do_predict(args)

