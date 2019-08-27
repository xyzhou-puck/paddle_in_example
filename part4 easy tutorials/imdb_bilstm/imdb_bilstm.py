import re
import numpy as np
import random
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.layers import basic_lstm as basic_lstm
from paddle.fluid.param_attr import ParamAttr


TRAIN_POS_PATTERN = re.compile("aclImdb/train/pos/.*\.txt$")
TRAIN_NEG_PATTERN = re.compile("aclImdb/train/neg/.*\.txt$")
TRAIN_PATTERN = re.compile("aclImdb/train/.*\.txt$")

TEST_POS_PATTERN = re.compile("aclImdb/test/pos/.*\.txt$")
TEST_NEG_PATTERN = re.compile("aclImdb/test/neg/.*\.txt$")
TEST_PATTERN = re.compile("aclImdb/test/.*\.txt$")

cutoff = 2
max_len = 100
batch_size = 32
embedding_dims = 128
lstm_hidden_size = 64
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
    batch_re_text = []
    batch_label = []
    batch_text_len = []

    for _ in range(epoch):
        
        random.shuffle(all_data)

        for item in all_data:
            text = item[0]
            label = item[1]

            if len(text) >= max_len:
                text = text[0:max_len]
                re_text = [text[max_len - i -1] for i in range(max_len)]
                batch_text_len.append(max_len)
            else:
                pad = [0] * (max_len - len(text))
                re_text = [text[len(text) - i -1] for i in range(len(text))]

                batch_text_len.append(len(text))
                text.extend(pad)
                re_text.extend(pad)

            batch_text.append(text)
            batch_label.append(label)
            batch_re_text.append(re_text)

            if len(batch_text) >= batch_size:
                yield np.array(batch_text).reshape((-1, max_len, 1)).astype("int64"), \
                    np.array(batch_re_text).reshape((-1, max_len, 1)).astype("int64"), \
                    np.array(batch_label).reshape((-1, 1)).astype("int64"), np.array(batch_text_len).astype("int32")
                batch_text = []
                batch_re_text = []
                batch_label = []
                batch_text_len = []

    if len(batch_text) > 0:
        yield np.array(batch_text).reshape((-1, max_len, 1)).astype("int64"), \
            np.array(batch_re_text).reshape((-1, max_len, 1)).astype("int64"), \
            np.array(batch_label).reshape((-1, 1)).astype("int64"), np.array(batch_text_len).astype("int32")
        batch_text = []
        batch_re_text = []
        batch_label = []
        batch_text_len = []



def build_model(is_training):
    
    input_text = fluid.layers.data(name = "text", shape = [-1, max_len, 1], dtype = "int64")
    input_re_text = fluid.layers.data(name = "re_text", shape = [-1, max_len, 1], dtype = "int64")

    input_text_len = fluid.layers.data(name = "text_len", shape = [-1], dtype = "int32")
    if is_training:
        input_label = fluid.layers.data(name = "label", shape = [-1, 1], dtype = "int64")

    input_text_emb = fluid.layers.embedding(input = input_text, size = [vocab_size, embedding_dims], param_attr = ParamAttr(name = "shared_emb"))
    input_re_text_emb = fluid.layers.embedding(input = input_re_text, size = [vocab_size, embedding_dims], param_attr = ParamAttr(name = "shared_emb"))

    _, _, input_text_lstm = basic_lstm(input_text_emb, None, None, lstm_hidden_size, num_layers = 1, sequence_length = input_text_len)

    _, _, input_re_text_lstm = basic_lstm(input_re_text_emb, None, None, lstm_hidden_size, num_layers = 1, sequence_length = input_text_len)

    input_text_lstm = fluid.layers.transpose(input_text_lstm, perm = [1, 0, 2])
    input_re_text_lstm = fluid.layers.transpose(input_re_text_lstm, perm = [1, 0, 2])

    input_text_lstm = fluid.layers.reshape(input_text_lstm, shape = [-1, lstm_hidden_size])
    input_re_text_lstm = fluid.layers.reshape(input_re_text_lstm, shape = [-1, lstm_hidden_size])

    input_text_hidden = fluid.layers.concat([input_text_lstm, input_re_text_lstm], axis = -1)

    input_text_hidden = fluid.layers.dropout(input_text_lstm, 0.5, is_test = not is_training)

    input_text_hidden = fluid.layers.fc(input_text_hidden, size = 2, act = "softmax")

    if is_training:
        loss = fluid.layers.cross_entropy(input_text_hidden, input_label)
        loss = fluid.layers.reduce_mean(loss)

        optimizer = fluid.optimizer.AdamOptimizer(learning_rate = 0.01)
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
for in_text, in_re_text, in_label, in_len in build_batch(batch_size, max_len, epochs, train_reader):

    out = exe.run(program = train_program,
        feed = {"text": in_text, "label": in_label, "text_len": in_len, "re_text": in_re_text},
        fetch_list = [loss.name])

    print("step %d, loss %.5f" % (step, out[0][0]))
    step += 1

