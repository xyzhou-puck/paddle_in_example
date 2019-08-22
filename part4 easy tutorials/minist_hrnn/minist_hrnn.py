import paddle.fluid as fluid
from paddle.dataset import mnist
from paddle.fluid.contrib.layers import basic_lstm
import paddle
import numpy as np
import os 

batch_size = 32 
num_classes = 10
epochs = 5

def batch_generator(generator, batch_size, epochs):

    batch_img = []
    batch_label = []
    for _ in range(epochs):
        for sample in generator():
            batch_img.append(np.reshape(sample[0], [28, 28]))
            batch_label.append([sample[1]])

            if len(batch_img) >= batch_size:
                yield np.array(batch_img).astype("float32"), np.array(batch_label).astype("int64")
                batch_img = []
                batch_label = []

    if batch_img:
        yield np.array(batch_img).astype("float32"), np.array(batch_label).astype("int64")


# define network
data = fluid.layers.data(name="img", shape=[-1, 28, 28], dtype='float32')
label = fluid.layers.data(name="label", shape=[-1,1], dtype='int64')
sequence_length = fluid.layers.data(name="sequence_length", shape=[-1], dtype='int32')
output_row, _, _ = basic_lstm(data, None, None, 128,sequence_length=sequence_length)
output_col, _, _ = basic_lstm(output_row, None, None, 128,sequence_length=sequence_length)
predict=fluid.layers.fc(input=output_row, size=num_classes,act="softmax")
cost = fluid.layers.cross_entropy(input=predict, label=label)
loss = fluid.layers.reduce_mean(cost)
acc = fluid.layers.accuracy(input = predict, label = label)


#set train and test program
test_program = fluid.default_main_program().clone(for_test=True)

#define optimizer
optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=0.001,rho=0.9)
optimizer.minimize(loss)


# use gpu or not
use_cuda = True
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

#define sequece_length for basic_lstm op
seq_lens = np.zeros(batch_size)
seq_lens[:] = 28 


#start training
step = 0
for batch_img, batch_label in batch_generator(mnist.train(), batch_size, epochs):
    step += 1
    out_loss,out_acc = exe.run(feed = {"img": batch_img, "label":batch_label, "sequence_length": seq_lens}, fetch_list = [loss.name,acc.name])
    if step % 100 == 0:
        print("step %d, loss %.3f, acc %.2f," % (step, out_loss[0],out_acc[0]))

# start testing
accuracy = fluid.metrics.Accuracy()
for batch_img, batch_label  in batch_generator(mnist.test(), batch_size, 1):
    out_pred = exe.run(program = test_program, feed = {"img": batch_img, "label":batch_label, "sequence_length": seq_lens}, fetch_list = [acc.name])
    accuracy.update(value = out_pred[0], weight = len(batch_img))
print("test acc: %.3f" % accuracy.eval())
