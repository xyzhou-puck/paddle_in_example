'''
Trains a DenseNet-40-12 model on the CIFAR-10 Dataset.

'''
import paddle.fluid as fluid
import paddle
import numpy
import os 
from PIL import Image
from densenet import DenseNet121

batch_size = 64
num_classes = 10
epochs = 300

# create the model (without loading weights)
data = fluid.layers.data(name="img", shape=[-1, 3, 32, 32], dtype='float32')
label = fluid.layers.data(name="label", shape=[-1, 1], dtype='int64')

mod = DenseNet121()
predict = mod.net(input=data, class_dim=num_classes)
cost = fluid.layers.softmax_with_cross_entropy(logits=predict, label=label)
loss = fluid.layers.reduce_mean(cost)
acc = fluid.layers.accuracy(input=predict, label=label)

test_program = fluid.default_main_program().clone(for_test=True)

optimizer = fluid.optimizer.SGD(
    learning_rate=fluid.layers.piecewise_decay(
        boundaries=[150, 225], values=[0.1, 0.01, 0.001]))
optimizer.minimize(loss)

place = fluid.CUDAPlace(1)
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

feeder = fluid.DataFeeder(feed_list=[data, label], place=place)

# dataset
train_dataset = paddle.dataset.cifar.train10()
test_dataset = paddle.dataset.cifar.test10()

# normalization
def normalization(generator):
    x, y = [], []
    for batch_img, batch_label in generator():
        x.append(batch_img)
        y.append(batch_label)
    x, y = numpy.array(x), numpy.array(y)

    x = numpy.reshape(x, (-1, 3, 32, 32))
    for i in range(3):
        x_mean = numpy.mean(x[:, i, :, :])
        x_std = numpy.std(x[:, i, :, :])
        x[:, i, :, :] = (x[:, i, :, :] - x_mean) / x_std
    x = numpy.reshape(x, (-1, 3072))

    def reader():
        for input, label in zip(x, y):
            yield input, label
    return reader

train_dataset = normalization(train_dataset)
test_dataset = normalization(test_dataset)

# start training
train_reader = paddle.batch(train_dataset, batch_size=batch_size)
test_reader = paddle.batch(test_dataset, batch_size=batch_size)

for epoch in range(epochs):
    mean_loss, mean_acc = [], []
    for step, data in enumerate(test_reader()):
        out = exe.run(feed = feeder.feed(data), fetch_list = [loss.name, acc.name])
        mean_loss.append(float(out[0]))
        mean_acc.append(float(out[1]))
    mean_loss = numpy.array(mean_loss).mean()
    mean_acc = numpy.array(mean_acc).mean()
    print("train loss: %.6f acc: %.3f" % (mean_loss, mean_acc))

# start testing
accuracy = fluid.metrics.Accuracy()
for batch_img, batch_label in enumerate(test_reader()):
    out_pred = exe.run(program = test_program, 
        feed = {"img": batch_img, "label":batch_label}, fetch_list = [acc.name])
    accuracy.update(value = out_pred[0], weight = len(batch_img))
print("test acc: %.3f" % accuracy.eval())

