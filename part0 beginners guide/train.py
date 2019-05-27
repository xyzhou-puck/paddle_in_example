#   Copyright (c) 2018 PaddlePaddle Authors. All Rights Reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

from __future__ import print_function

import sys

import math
import numpy

import paddle
import paddle.fluid as fluid

def save_result(points1, points2):
    """
    Save the  results into a picture.
    """
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    x1 = [idx for idx in range(len(points1))]
    y1 = points1
    y2 = points2
    l1 = plt.plot(x1, y1, 'r--', label='predictions')
    l2 = plt.plot(x1, y2, 'g--', label='GT')
    plt.plot(x1, y1, 'ro-', x1, y2, 'g+-')
    plt.title('predictions VS GT')
    plt.legend()
    plt.savefig('./image/prediction_gt.png')


def get_next_batch(data, batch_size = 20):
    """
    Genrate next batch for the given reader
    reader: an iterable function, producing examples for training/testing
    batch_size: the batch size for training/testing
    """
    idx = 0
    batch_x = []
    batch_y = []

    while True:

        if idx >= len(data):
            idx = 0

        batch_x.append(data[idx][:-1])
        batch_y.append(data[idx][-1:])

        idx += 1
        if len(batch_x) >= batch_size:
            yield numpy.array(batch_x).astype("float32"), numpy.array(batch_y).astype("float32")
            batch_x = []
            batch_y = []


def main():
    """
    The main function for fitting a line.
    """

    # download data for training and testing
    # the training data is stored at paddle.dataset.uci_housing.UCI_TRAIN_DATA
    # the testing data is stored at paddle.dataset.uci_housing.UCI_TEST_DATA
    URL = 'http://paddlemodels.bj.bcebos.com/uci_housing/housing.data'
    MD5 = 'd4accdce7a25600298819f8e28e8d593'
    paddle.dataset.uci_housing.load_data(paddle.dataset.common.download(URL, 'uci_housing', MD5))

    batch_size = 20
    batch_num = len(paddle.dataset.uci_housing.UCI_TRAIN_DATA) // batch_size
    num_epochs = 100
    
    # can use CPU or GPU
    use_cuda = False

    # Specify the directory to save the parameters
    params_dirname = "fit_a_line.inference.model"
    train_prompt = "Train cost"
    test_prompt = "Test cost"

    # define the graph here
    # feature vector of length 13
    # the place holder for features (x)
    x = fluid.layers.data(name='x', shape=[-1, 13], dtype='float32')
    # the place holder for true house price y
    y = fluid.layers.data(name='y', shape=[-1, 1], dtype='float32')
    # the predicted price y_predict
    y_predict = fluid.layers.fc(input=x, size=1, act=None)

    # define the loss function here
    cost = fluid.layers.square_error_cost(input=y_predict, label=y)
    avg_loss = fluid.layers.mean(cost)

    # define the optimizer here, using SGDOptimizer
    sgd_optimizer = fluid.optimizer.SGDOptimizer(learning_rate=0.001)
    sgd_optimizer.minimize(avg_loss)

    main_program = fluid.default_main_program()
    startup_program = fluid.default_startup_program()
    test_program = main_program.clone(for_test=True)

    place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
    exe = fluid.Executor(place)
    
    # init all variables before starting training
    exe.run(startup_program)
    # now start training
    for pass_id in range(num_epochs):
        for step_id in range(batch_num):

            train_x, train_y = next(get_next_batch(paddle.dataset.uci_housing.UCI_TRAIN_DATA, batch_size))
    
            avg_loss_value, = exe.run(
                program = main_program,
                feed = {"x" : train_x, 
                        "y" : train_y},
                fetch_list=[avg_loss])
            
            if step_id % 10 == 0:  # record a train cost every 10 batches
                print("%s, Pass %d, Step %d, Cost %f" %
                      (train_prompt, pass_id, step_id, avg_loss_value[0]))

            if math.isnan(float(avg_loss_value[0])):
                sys.exit("got NaN loss, training failed.")
        
        if params_dirname is not None:
            # We can save the trained parameters for the inferences later
            fluid.io.save_inference_model(params_dirname, ['x'], [y_predict],
                                          exe)

    
    # now start infering
    infer_exe = fluid.Executor(place)
    inference_scope = fluid.core.Scope()

    # infer
    with fluid.scope_guard(inference_scope):
        [inference_program, feed_target_names, fetch_targets
         ] = fluid.io.load_inference_model(params_dirname, infer_exe)
        batch_size = 10

        infer_x, infer_y = next(get_next_batch(paddle.dataset.uci_housing.UCI_TEST_DATA, batch_size))
        infer_feat = numpy.array( infer_x ).astype("float32")
        infer_label = numpy.array( infer_y ).astype("float32")

        assert feed_target_names[0] == 'x'
        
        results = infer_exe.run(
            inference_program,
            feed={feed_target_names[0]: infer_feat},
            fetch_list=fetch_targets)

        print("infer results: (House Price)")
        for idx, val in enumerate(results[0]):
            print("%d: %.2f" % (idx, val))

        print("\nground truth:")
        for idx, val in enumerate(infer_label):
            print("%d: %.2f" % (idx, val))

        save_result(results[0], infer_label)


if __name__ == '__main__':
    main()
