import re
import numpy as np
import random
import math
import pandas as pd
import matplotlib.pyplot as plt
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.contrib.layers import basic_lstm as basic_lstm
from paddle.fluid.param_attr import ParamAttr

tsteps = 1
lahead = 1
batch_size = 25
epochs = 25

def gen_cosine_amp(amp=100, 
                   period=1000, 
                   x0=0, 
                   xn=50000, 
                   step=1, 
                   k=0.0001):
    n = (xn - x0) * step
    cos = np.full((n, 1, 1), n).astype('float32')
    for i in range(len(cos)):
        idx = x0 + i * step
        cos[i, 0, 0] = amp * math.cos(2 * math.pi * idx / period)
        cos[i] *= math.exp(-k * idx)
    return cos

print('Generating Data...\n')
cos = gen_cosine_amp()
print('Input shape:', cos.shape)

expected_output = np.full((len(cos), 1), len(cos)).astype('float32')
for i in range(len(cos)):
    expected_output[i, 0] = np.mean(cos[i : i + lahead])

print('Output shape:', expected_output.shape)

print('Creating model:\n')

x = layers.data(name="x", shape=[-1, tsteps, 1], dtype="float32")
y = layers.data(name="y", shape=[-1, 1], dtype="float32")

lstm1_init_h = layers.data(name="lstm1_h", shape=[1, batch_size, 50], 
    dtype="float32", append_batch_size=False)
lstm1_init_c = layers.data(name="lstm1_c", shape=[1, batch_size, 50], 
    dtype="float32", append_batch_size=False)

lstm1, lstm1_h, lstm1_c = basic_lstm(x, lstm1_init_h, lstm1_init_c, 50, num_layers=1)
_, lstm2_h, lstm2_c = basic_lstm(lstm1, lstm1_h, lstm1_c, 50, num_layers=1)
lstm2_c_batch_first = layers.transpose(lstm2_c, [1, 0, 2])
pred = layers.fc(lstm2_c_batch_first, 1)
loss = layers.reduce_mean(layers.square(pred - y))

test_program = fluid.default_main_program().clone(for_test=True)

optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=0.001)
optimizer.minimize(loss)

exe = fluid.Executor(fluid.CPUPlace())
exe.run(fluid.default_startup_program())

def batch_generator(x_data, y_data, batch_size):
    batch_x, batch_y = [], []
    for sample_x, sample_y in zip(x_data, y_data):
        batch_x.append(sample_x)
        batch_y.append(sample_y)

        if len(batch_x) >= batch_size:
            yield np.array(batch_x).astype("float32"), \
                np.array(batch_y).astype("float32")
            batch_x, batch_y = [], []

lstm1_init_h_np = np.zeros((1, batch_size, 50)).astype('float32')
lstm1_init_c_np = np.zeros((1, batch_size, 50)).astype('float32')

print('Training\n')
for epoch in range(epochs):
    step = 0
    mean_loss = []
    for batch_x, batch_y in batch_generator(cos, expected_output, batch_size):
        out = exe.run(
            feed = {"x": batch_x, 
                    "y": batch_y, 
                    "lstm1_h": lstm1_init_h_np,
                    "lstm1_c": lstm1_init_c_np},
            fetch_list = [loss.name, lstm2_h.name, lstm2_c.name])
        mean_loss.append(out[0])
        lstm1_init_h_np = out[1]
        lstm1_init_c_np = out[2]

        step += 1
        if (step % 200 == 0):
            print("step %d loss: %f" % (step, np.array(mean_loss).mean()))

    # reset states
    lstm1_init_h_np = np.zeros((1, batch_size, 50)).astype('float32')
    lstm1_init_c_np = np.zeros((1, batch_size, 50)).astype('float32')

    mean_loss = np.array(mean_loss).mean()
    print("epoch: %d mean loss: %f" % (epoch, mean_loss))

print('Predicting\n')

# reset states
lstm1_init_h_np = np.zeros((1, batch_size, 50)).astype('float32')
lstm1_init_c_np = np.zeros((1, batch_size, 50)).astype('float32')

stateless_pred_list, mean_loss = [], []
for batch_x, batch_y in batch_generator(cos, expected_output, batch_size):
    out = exe.run(
        program = test_program,
        feed = {"x": batch_x, 
                "y": batch_y, 
                "lstm1_h": lstm1_init_h_np,
                "lstm1_c": lstm1_init_c_np},
        fetch_list = [loss.name, pred.name])
    mean_loss.append(out[0])
    stateless_pred_list.append(out[1])
mean_loss = np.array(mean_loss).mean()
stateless_pred_list = np.concatenate(stateless_pred_list)
print("testing stateless: mean loss: ", mean_loss)

# reset states
lstm1_init_h_np = np.zeros((1, batch_size, 50)).astype('float32')
lstm1_init_c_np = np.zeros((1, batch_size, 50)).astype('float32')

stateful_pred_list, mean_loss = [], []
for batch_x, batch_y in batch_generator(cos, expected_output, batch_size):
    out = exe.run(
        program = test_program,
        feed = {"x": batch_x, 
                "y": batch_y, 
                "lstm1_h": lstm1_init_h_np,
                "lstm1_c": lstm1_init_c_np},
        fetch_list = [loss.name, lstm2_h.name, lstm2_c.name, pred.name])
    lstm1_init_h_np = out[1]
    lstm1_init_c_np = out[2]
    mean_loss.append(out[0])
    stateful_pred_list.append(out[3])
mean_loss = np.array(mean_loss).mean()
stateful_pred_list = np.concatenate(stateful_pred_list)
print("testing stateful: mean loss: ", mean_loss)

plt.subplot(3, 1, 1)
plt.plot(expected_output)
plt.title('Expected')
plt.subplot(3, 1, 2)
plt.plot(stateless_pred_list, label='pred')
plt.plot((expected_output - stateless_pred_list).flatten(), label='err')
plt.title('Stateless')
plt.subplot(3, 1, 3)
plt.plot(stateful_pred_list, label='pred')
plt.plot((expected_output - stateful_pred_list).flatten(), label='err')
plt.title('Stateful')
plt.show()
