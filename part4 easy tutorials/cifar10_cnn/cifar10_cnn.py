import paddle.fluid as fluid
import paddle
import numpy
import os 

batch_size = 32 
num_classes = 10
epochs = 100

# define data reader
def batch_generator(generator, batch_size, epochs):
    batch_img = []
    batch_label = []
    for _ in range(epochs):
        for sample in generator():
            batch_img.append(numpy.reshape(sample[0], [32, 32,3]))
            batch_label.append([sample[1]])

            if len(batch_img) >= batch_size:
                yield numpy.array(batch_img).astype("float32"), numpy.array(batch_label).astype("int64")
                batch_img = []
                batch_label = []
    if batch_img:
        yield numpy.array(batch_img).astype("float32"), numpy.array(batch_label).astype("int64")

# define network
data = fluid.layers.data(name="img", shape=[32, 32, 3], dtype='float32')
label = fluid.layers.data(name="label", shape=[1], dtype='int64')
conv2d1 = fluid.layers.conv2d(input=data, num_filters=32, padding=(3,3),filter_size=(3,3), act="relu")
conv2d2 = fluid.layers.conv2d(input=conv2d1, num_filters=32,filter_size=(3,3), act="relu")
pool2d2 = fluid.layers.pool2d(input=conv2d2,pool_size=2,pool_type='max')
droped = fluid.layers.dropout(pool2d2, dropout_prob=0.25)
conv2d3 = fluid.layers.conv2d(input=droped, num_filters=64, padding=(3,3),filter_size=3, act="relu")
conv2d4 = fluid.layers.conv2d(input=conv2d3, num_filters=64, filter_size=3, act="relu")
pool2d3 = fluid.layers.pool2d(input=conv2d4,pool_size=2,pool_type='max')
droped3 = fluid.layers.dropout(pool2d3, dropout_prob=0.25)
flatten3=fluid.layers.flatten(droped3, axis=1)
fc3=fluid.layers.fc(input=flatten3, size=512,act="relu")
droped4 = fluid.layers.dropout(fc3, dropout_prob=0.5)
predict=fluid.layers.fc(input=droped4, size=num_classes,act="softmax")
cost = fluid.layers.cross_entropy(input=predict, label=label)
loss = fluid.layers.reduce_mean(cost)
acc = fluid.layers.accuracy(input = predict, label = label)

#set train and test program
test_program = fluid.default_main_program().clone(for_test=True)

#define optimizer
optimizer = fluid.optimizer.RMSPropOptimizer(learning_rate=0.001,         regularization=fluid.regularizer.L2DecayRegularizer(regularization_coeff=0.0000001))
optimizer.minimize(loss)


# use gpu or not
use_cuda = False
place = fluid.CUDAPlace(0) if use_cuda else fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

#start training
step = 0
for batch_img, batch_label in batch_generator(paddle.dataset.cifar.train10(), batch_size, epochs):
    step += 1
    out_loss = exe.run(feed = {"img": batch_img, "label":batch_label}, fetch_list = [loss.name])
    if step % 100 == 0:
        print("step %d, loss %.3f" % (step, out_loss[0]))

# start testing
accuracy = fluid.metrics.Accuracy()
for batch_img, batch_label in batch_generator(paddle.dataset.cifar.test10(), batch_size, 1):
    out_pred = exe.run(program = test_program, feed = {"img": batch_img, "label":batch_label}, fetch_list = [acc.name])
    accuracy.update(value = out_pred[0], weight = len(batch_img))
print("test acc: %.3f" % accuracy.eval())

