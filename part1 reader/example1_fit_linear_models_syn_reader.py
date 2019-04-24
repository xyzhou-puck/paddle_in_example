#encoding=utf8

import os
import sys
import numpy as np
import paddle
import paddle.fluid as fluid

import synchronic_reader

# set that random seed
true_factors = [4, 6, 7, 2]

# define the reader

batch_reader = synchronic_reader.batch_reader_creator(true_factors, 100)

# define the network
# like the placeholder in tf
x = fluid.layers.data(name="x", shape=[4], dtype="float32")
y = fluid.layers.data(name="y", shape=[1], dtype="float32")
y_predict = fluid.layers.fc(input=x, size =1, act=None)

# define the loss function
cost = fluid.layers.square_error_cost(input=y_predict, label=y)
avg_cost = fluid.layers.mean(cost)

# define the optimizer
sgd_optimizer = fluid.optimizer.SGD(learning_rate = 0.05)
sgd_optimizer.minimize(avg_cost)

# init that parameters
cpu = fluid.CPUPlace()
exe = fluid.Executor(cpu)
exe.run(fluid.default_startup_program())

# start training
for step in xrange(500):
    
    train_data, y_true = next(batch_reader())

    train_data = np.array(train_data).astype('float32') 
    y_true = np.array(y_true).astype('float32')

    outs = exe.run(
            feed = {'x':train_data, 'y':y_true},
            fetch_list = [y_predict.name, avg_cost.name])

    # print out the loss for every 50 steps
    if step % 50 == 0:
        print('iter={:.0f}, cost={}'.format(step, outs[1][0]))


sample_reader = synchronic_reader.sample_reader_creator(true_factors)
test_data, test_true_y = next(sample_reader())

# store the training results
params_dirname = "result"
fluid.io.save_inference_model(params_dirname, ['x'], [y_predict], exe)

# start inference
infer_exe = fluid.Executor(cpu)
inference_scope = fluid.Scope()

# load that saved model
with fluid.scope_guard(inference_scope):
    [inference_program, feed_target_names, 
            fetch_targets] = fluid.io.load_inference_model(params_dirname, infer_exe)


# generate some test data
sample_reader = synchronic_reader.sample_reader_creator(true_factors)

test_data, test_true_y = next(sample_reader())

test_data = np.array(test_data).astype('float32')
test_true_y = np.array(test_true_y).astype('float32')

# infer outptus for given test data
results = infer_exe.run(inference_program,
                        feed = {"x": test_data},
                        fetch_list = fetch_targets)

# print the output of inference
print("4a+6b+7c+2d={}, expected={}".format(results[0][0], test_true_y[0]))


