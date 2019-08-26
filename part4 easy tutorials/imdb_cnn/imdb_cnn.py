import re
import numpy as np
import random
import paddle
import paddle.fluid as fluid

TRAIN_POS_PATTERN = re.compile("aclImdb/train/pos/.*\.txt$")
TRAIN_NEG_PATTERN = re.compile("aclImdb/train/neg/.*\.txt$")
TRAIN_PATTERN = re.compile("aclImdb/train/.*\.txt$")

TEST_POS_PATTERN = re.compile("aclImdb/test/pos/.*\.txt$")
TEST_NEG_PATTERN = re.compile("aclImdb/test/neg/.*\.txt$")
TEST_PATTERN = re.compile("aclImdb/test/.*\.txt$")

cutoff = 2
max_len = 400
batch_size = 32
embedding_dims = 50
filters = 250
kernel_size = 3
hidden_dims = 250
epochs = 2

word_idx = paddle.dataset.imdb.build_dict(TRAIN_PATTERN, cutoff)

vocab_size = len(word_idx) + 1

train_reader = paddle.dataset.imdb.train(word_idx)
test_reader = paddle.dataset.imdb.test(word_idx)

def build_batch(batch_size, max_len, epoch, reader):
    
    all_data = []
    for item in reader():
        all_data.append(item)

    batch_text = []
    batch_label = []

    for _ in range(epoch):
        
        random.shuffle(all_data)

        for item in all_data:
            text = item[0]
            label = item[1]

            if len(text) >= max_len:
                text = text[0:max_len]
            else:
                pad = [0] * (max_len - len(text))
                pad.extend(text)
                text = pad

            batch_text.append(text)
            batch_label.append(label)

            if len(batch_text) >= batch_size:
                yield np.array(batch_text).reshape((-1, max_len, 1)).astype("int64"), np.array(batch_label).reshape((-1, 1)).astype("int64")
                batch_text = []
                batch_label = []

    if len(batch_text) > 0:
        yield np.array(batch_text).reshape((-1, max_len, 1)).astype("int64"), np.array(batch_label).reshape((-1, 1)).astype("int64")
        batch_text = []
        batch_label = []


def build_model(is_training):
    
    input_text = fluid.layers.data(name = "text", shape = [-1, max_len, 1], dtype = "int64")
    if is_training:
        input_label = fluid.layers.data(name = "label", shape = [-1, 1], dtype = "int64")

    input_text_emb = fluid.layers.embedding(input = input_text, size = [vocab_size, embedding_dims])

    input_text_emb = fluid.layers.transpose(input_text_emb, perm = [0, 2, 1])
    input_text_emb = fluid.layers.reshape(input_text_emb, shape = [-1, embedding_dims, max_len, 1])

    input_text_emb = fluid.layers.dropout(input_text_emb, 0.2, is_test = not is_training)

    input_text_conv = fluid.layers.conv2d(input = input_text_emb, num_filters = filters, filter_size = (kernel_size, 1), stride = (1, 1))
    input_text_conv = fluid.layers.relu(input_text_conv)

    input_text_conv = fluid.layers.pool2d(input_text_conv, pool_size = (max_len - kernel_size + 1, 1))

    input_text_hidden = fluid.layers.reshape(input_text_conv, shape = [-1, filters])

    input_text_hidden = fluid.layers.dropout(input_text_hidden, 0.2, is_test = not is_training)
    input_text_hidden = fluid.layers.relu(input_text_hidden)

    input_text_hidden = fluid.layers.fc(input_text_hidden, size = 2, act = "softmax")

    if is_training:
        loss = fluid.layers.cross_entropy(input_text_hidden, input_label)
        loss = fluid.layers.reduce_mean(loss)

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate = 0.001)
        optimizer.minimize(loss)

        return loss

    else:

        return input_text_hidden

startup_program = fluid.Program()
train_program = fluid.Program()
test_program = fluid.Program()

with fluid.program_guard(train_program, startup_program):
    with fluid.unique_name.guard():
        loss = build_model(is_training = True)

exe = fluid.Executor(fluid.CPUPlace())

exe.run(startup_program)
step = 0
for in_text, in_label in build_batch(batch_size, max_len, epochs, train_reader):
    out = exe.run(program = train_program,
        feed = {"text": in_text, "label": in_label},
        fetch_list = [loss.name])

    print("step %d, loss %.5f" % (step, out[0][0]))
    step += 1

