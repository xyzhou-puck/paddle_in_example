#encoding=utf8

import os
import sys
import numpy as np
import paddle
import paddle.fluid as fluid

import asynchronic_reader as asynchronic_reader

train_main_prog = fluid.Program()
test_main_prog = fluid.Program()
startup_prog = fluid.Program()

# set that random seed
true_factors = [4, 6, 7, 2]

with fluid.program_guard(train_main_prog, startup_prog):
    with fluid.unique_name.guard("model"):

        x = fluid.layers.data(name="X", shape=[-1, 4], dtype="float32")
        y = fluid.layers.data(name="Y", shape=[-1, 1], dtype="float32")

        asyn_reader = fluid.layers.create_py_reader_by_data(capacity = 1000, feed_list=[x, y], name="train_reader")
        batch_reader = asynchronic_reader.create_batch_reader(true_factors, batch_size = 100)
        asyn_reader.decorate_paddle_reader(batch_reader)

        x, y = fluid.layers.read_file(asyn_reader)

        y_predict = fluid.layers.fc(input=x, size =1, act=None)
        
        cost = fluid.layers.square_error_cost(input=y_predict, label=y)
        avg_cost = fluid.layers.mean(cost)

        sgd_optimizer = fluid.optimizer.SGD(learning_rate = 0.05)
        sgd_optimizer.minimize(avg_cost)

# share the parameters from train_main_prog
with fluid.program_guard(test_main_prog, startup_prog):
    with fluid.unique_name.guard("model"):
        x = fluid.layers.data(name="X", shape=[-1, 4], dtype="float32")
        y = fluid.layers.data(name="Y", shape=[-1, 1], dtype="float32")

        asyn_test_reader = fluid.layers.create_py_reader_by_data(capacity = 1000, feed_list=[x, y], name="test_reader")
        
        batch_test_reader = asynchronic_reader.create_batch_reader(true_factors, batch_size = 1)
        asyn_test_reader.decorate_paddle_reader(batch_test_reader)

        x, y = fluid.layers.read_file(asyn_test_reader)

        y_predict = fluid.layers.fc(input=x, size =1, act=None)

_place = fluid.CUDAPlace(0)
exe = fluid.Executor(_place)
exe.run(program = startup_prog)

train_exe = fluid.Executor(_place)
asyn_reader.start()

step = 0
while True:
    outs = train_exe.run( program = train_main_prog,
        fetch_list = [y_predict.name, avg_cost.name])
    if step % 50 == 0:
        print('iter={:.0f}, cost={}'.format(step, outs[1][0]))

    step += 1
    if step >= 1000:
        break

# store the training results
params_dirname = "result"
fluid.io.save_inference_model(params_dirname, [x.name], [y_predict], train_exe, main_program = train_main_prog)

# start inferece
test_exe = fluid.Executor(_place)

asyn_test_reader.start()

outs = test_exe.run(program = test_main_prog,
        fetch_list = [y_predict.name, y.name])

print("4a+6b+7c+2d={}, expected={}".format(outs[0], outs[1]))

