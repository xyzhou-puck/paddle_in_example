import numpy
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.dataset import mnist

# hyper-parameters

batch_size = 128
num_classes = 10
epochs = 12

img_rows = 28
img_cols = 28

# define the model

X = layers.data(name = "img", shape = [-1, 1, 28, 28], dtype = "float32")
Y = layers.data(name = "label", shape = [-1, 1], dtype = "int64")

h_conv = layers.conv2d(X, num_filters = 32, filter_size = (3, 3), act = "relu")
h_conv = layers.conv2d(h_conv, num_filters = 64, filter_size = (3, 3), act = "relu")
h_pool = layers.pool2d(h_conv, pool_size = (2, 2))
h_dropout = layers.dropout(h_pool, dropout_prob=0.25)
h_flatten = layers.flatten(h_dropout)
h_fc = layers.fc(h_flatten, size = 128, act = "relu", 
    bias_attr=fluid.param_attr.ParamAttr(name="b_0"))
h_dropout2 = layers.dropout(h_fc, dropout_prob=0.25)
pred = layers.fc(h_dropout2, size = num_classes, act = "softmax",
    bias_attr=fluid.param_attr.ParamAttr(name="b_1"))

loss = layers.reduce_mean( layers.cross_entropy(input=pred, label=Y) )
acc = layers.accuracy(input = pred, label = Y)

test_program = fluid.default_main_program().clone(for_test=True)

# define the optimizer 
optimizer = fluid.optimizer.Adadelta(
    learning_rate=1.0, rho=0.95)

optimizer.minimize(loss)

# define the executor

exe = fluid.Executor(fluid.CUDAPlace(0))
exe.run(fluid.default_startup_program())

# define data reader

def batch_generator(generator, batch_size, epochs):

    batch_img = []
    batch_label = []
    for _ in range(epochs):
        for sample in generator():
            batch_img.append(numpy.reshape(sample[0], [1, img_rows, img_cols]))
            batch_label.append([sample[1]])

            if len(batch_img) >= batch_size:
                yield numpy.array(batch_img).astype("float32"), numpy.array(batch_label).astype("int64")
                batch_img = []
                batch_label = []

    if batch_img:
        yield numpy.array(batch_img).astype("float32"), numpy.array(batch_label).astype("int64")

# start training
step = 0
for batch_img, batch_label in batch_generator(mnist.train(), batch_size, epochs):
    step += 1
    out_loss = exe.run(feed = {"img": batch_img, "label":batch_label}, fetch_list = [loss.name])

    if step % 100 == 0:
        print("step %d, loss %.3f" % (step, out_loss[0]))


# start testing
accuracy = fluid.metrics.Accuracy()
for batch_img, batch_label in batch_generator(mnist.test(), batch_size, 1):
    out_pred = exe.run(program = test_program, feed = {"img": batch_img, "label":batch_label}, fetch_list = [acc.name])
    
    accuracy.update(value = out_pred[0], weight = len(batch_img))

print("test acc: %.3f" % accuracy.eval())

