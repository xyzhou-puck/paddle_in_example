import os, shutil
import re
import copy
import numpy as np
import random
import math
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.contrib.layers import basic_lstm, BasicLSTMUnit
from paddle.fluid.layers.control_flow import StaticRNN
from paddle.fluid.param_attr import ParamAttr
from progress.bar import Bar

# Preprocessing -----------------------------------------------------------

# Assumes you've downloaded and unzipped one of the bilingual datasets offered at
# http://www.manythings.org/anki/ and put it into a directory "data"
# This example translates English to Dutch.

samples = 10000
split = 0.8
filepath = './data/nld.txt'

with open(filepath, 'r') as f:
    lines = f.read().split('\n')[:samples]

def space_before_punct(sentence):
    return re.sub(r'([?.!])', ' \\1', sentence)

def replace_special_chars(sentence):
    return re.sub(r'[^a-zA-Z?.!,¿]+', ' ', sentence)

def add_token(sentence):
    return '<start> ' + sentence + ' <stop>'

def preprocess_sentence(line):
    line = space_before_punct(line)
    line = replace_special_chars(line)
    line = add_token(line)
    return line

def create_index(lines):
    words = set()
    for line in lines:
        for word in line.split(' '):
            if word not in words:
                words.add(word)
    words = sorted(list(words))
    words.insert(0, '<pad>')
    return words

def index2word(words):
    return { i : words[i] for i in range(len(words))}

def word2index(words):
    return { words[i] : i for i in range(len(words))}

def sentence2digits(sentence, word_dict):
    sentence_digits = []
    for word in sentence.split(' '):
        sentence_digits.append(word_dict[word])
    return sentence_digits

def sentlist2diglist(sentence_list, word_dict):
    diglist = []
    for sentence in sentence_list:
        diglist.append(sentence2digits(sentence, word_dict))
    return diglist

def pad_sequences(sentence_list, maxlen):
    padding_sentence_list = copy.deepcopy(sentence_list)
    mask = []
    for sentence in padding_sentence_list:
        sentence_mask = [1] * len(sentence)
        sentence_mask.extend([0] * (maxlen - len(sentence)))
        sentence.extend([0] * (maxlen - len(sentence)))
        mask.append(sentence_mask)
    return padding_sentence_list, mask

input_texts = []
target_texts = []
for line in lines:
    src, tar = line.split('\t')
    input_texts.append(preprocess_sentence(src))
    target_texts.append(preprocess_sentence(tar))
src_index = create_index(input_texts)
tar_index = create_index(target_texts)
src_word_dict = word2index(src_index)
tar_word_dict = word2index(tar_index)
src_idx_dict = index2word(src_index)
tar_idx_dict = index2word(tar_index)

src_diglist = sentlist2diglist(input_texts, src_word_dict)
src_maxlen = max({len(digits) for digits in src_diglist})
src_matrix, src_mask = pad_sequences(src_diglist, src_maxlen)

tar_diglist = sentlist2diglist(target_texts, tar_word_dict)
tar_maxlen = max({len(digits) for digits in tar_diglist})
tar_matrix, tar_mask = pad_sequences(tar_diglist, tar_maxlen)

# Train-test-split --------------------------------------------------------

seed = 11
train_size = math.floor(len(src_matrix) * split)
random.Random(seed).shuffle(src_matrix)
random.Random(seed).shuffle(tar_matrix)
random.Random(seed).shuffle(src_mask)
random.Random(seed).shuffle(tar_mask)

x_train = np.expand_dims(np.array(src_matrix[:train_size]).astype('int64'), 2)
y_train = np.expand_dims(np.array(tar_matrix[:train_size]).astype('int64'), 2)
x_valid = np.expand_dims(np.array(src_matrix[train_size:]).astype('int64'), 2)
y_valid = np.expand_dims(np.array(tar_matrix[train_size:]).astype('int64'), 2)
y_train_mask = np.expand_dims(np.array(tar_mask[:train_size]).astype('float32'), 2)
y_valid_mask = np.expand_dims(np.array(tar_mask[train_size:]).astype('float32'), 2)

buffer_size = len(x_train)

# just for convenience, so we may get a glimpse at translation performance 
# during training

train_sentences = x_train
validation_sentences = x_valid
validation_samples = validation_sentences[:5]

# Hyperparameters / variables ---------------------------------------------

batch_size = 32
embedding_dim = 64
lstm_units = 256

src_vocab_size = len(src_index)
tar_vocab_size = len(tar_index)

# Create datasets ---------------------------------------------------------

def batch_creator(x_data, y_data, y_mask, batch_size, drop_last=True):
    def batch_generator():
        batch_x, batch_y, batch_y_mask = [], [], []
        data = zip(x_data, y_data, y_mask)
        for x, y, ym in data:
            batch_x.append(x)
            batch_y.append(y)
            batch_y_mask.append(ym)

            if len(batch_x) >= batch_size:
                yield np.array(batch_x).astype("int64"), \
                    np.array(batch_y).astype("int64"), np.array(batch_y_mask).astype("float32")
                batch_x, batch_y, batch_y_mask = [], [], []
        if batch_x and not drop_last:
            yield np.array(batch_x).astype("int64"), \
                np.array(batch_y).astype("int64"), np.array(batch_y_mask).astype("float32")
    return batch_generator

train_dataset = batch_creator(x_train, y_train, y_train_mask, batch_size)
validation_dataset = batch_creator(x_valid, y_valid, y_valid_mask, batch_size)

# Attention encoder -------------------------------------------------------

def attention_encoder(input, 
                      init_hidden,
                      init_cell,
                      lstm_units, 
                      embedding_dim, 
                      src_vocab_size,
                      name='attention_encoder'):
    h_embedding = layers.embedding(
        input=input,
        size=[src_vocab_size, embedding_dim],
        dtype='float32')
    rnn_out, last_hidden, last_cell = basic_lstm(
        input=h_embedding,
        init_cell=init_cell,
        init_hidden=init_hidden,
        hidden_size=lstm_units)
    return rnn_out, last_hidden, last_cell

# Attention decoder -------------------------------------------------------

def attention_decoder(dec_input,
                      enc_last_hidden,
                      enc_last_cell,
                      encoder_output,
                      lstm_units,
                      embedding_dim,
                      tar_vocab_size,
                      name='attention_decoder'):

    dec_unit = BasicLSTMUnit("dec_layer", lstm_units, dtype='float32')

    w_1 = layers.create_parameter(
        shape=[lstm_units, lstm_units], 
        dtype="float32", 
        name=name + "_w1", 
        default_initializer=fluid.initializer.Xavier())

    w_2 = layers.create_parameter(
        shape=[lstm_units, lstm_units], 
        dtype="float32", 
        name=name + "_w2", 
        default_initializer=fluid.initializer.Xavier())

    w_v = layers.create_parameter(
        shape=[lstm_units, 1], 
        dtype="float32", 
        name=name + "_wv", 
        default_initializer=fluid.initializer.Xavier())

    dec_input = layers.embedding(
        input=dec_input,
        size=[tar_vocab_size, embedding_dim],
        dtype='float32',
        param_attr=ParamAttr(name=name + '_emb'))

    dec_input = layers.transpose(dec_input, (1, 0, 2))

    dec_rnn = StaticRNN()
    with dec_rnn.step():

        step_input = dec_rnn.step_input(dec_input)

        pre_hidden = dec_rnn.memory(init=enc_last_hidden[0])
        pre_cell = dec_rnn.memory(init=enc_last_cell[0])

        hidden_with_time_axis = layers.unsqueeze(pre_hidden, axes=[1])
        hidden_with_time_axis = layers.expand(hidden_with_time_axis, [1, src_maxlen, 1])

        w1 = layers.matmul(encoder_output, w_1)
        w2 = layers.matmul(hidden_with_time_axis, w_2)
        score = layers.matmul(layers.tanh(w1 + w2), w_v)
        attention_weights = layers.softmax(score, axis=1)
        
        context_vector = layers.elementwise_mul(encoder_output, attention_weights, axis=0)
        context_vector = layers.reduce_sum(context_vector, dim=1)

        x = layers.concat([context_vector, step_input], -1)

        new_hidden, new_cell = dec_unit(x, pre_hidden, pre_cell)

        dec_rnn.update_memory(pre_hidden, new_hidden)
        dec_rnn.update_memory(pre_cell, new_cell)

        dec_rnn.step_output(new_hidden)
        dec_rnn.step_output(attention_weights)

    dec_rnn_out, attention_matrix = dec_rnn()

    dec_rnn_out = layers.transpose(dec_rnn_out, (1, 0, 2))

    dec_rnn_out = layers.fc(
        dec_rnn_out, 
        num_flatten_dims=2,
        size=tar_vocab_size, 
        param_attr=ParamAttr(name=name + '_fc4_w'),
        bias_attr=ParamAttr(name=name + '_fc4_b'))

    attention_matrix = layers.transpose(attention_matrix, (1, 0, 2, 3))

    return dec_rnn_out, attention_matrix

# The model ---------------------------------------------------------------

input = layers.data(
    name='input', 
    shape=[-1, src_maxlen, 1], 
    dtype='int64')

target_pop_last = layers.data(
    name='target_pop_last',
    shape=[-1, tar_maxlen - 1, 1], 
    dtype='int64')

target_pop_first = layers.data(
    name='target_pop_first',
    shape=[-1, tar_maxlen - 1, 1], 
    dtype='int64')

target_mask_pop_first = layers.data(
    name='target_mask_pop_first',
    shape=[-1, tar_maxlen - 1, 1], 
    dtype='float32')

init_hidden = layers.fill_constant_batch_size_like(input,
    shape=(-1, 1, lstm_units), value=0, dtype='float32')

init_hidden = layers.transpose(init_hidden, (1, 0, 2))

init_cell = layers.fill_constant_batch_size_like(input,
    shape=(-1, 1, lstm_units), value=0, dtype='float32')

init_cell = layers.transpose(init_cell, (1, 0, 2))

enc_output, enc_hidden, enc_cell = attention_encoder(input, init_hidden, init_cell,
    lstm_units, embedding_dim, src_vocab_size)

preds, attention_matrix = attention_decoder(target_pop_last, enc_hidden, enc_cell, enc_output,
    lstm_units, embedding_dim, tar_vocab_size)

preds_out = layers.softmax(preds)

test_program = fluid.default_main_program().clone(for_test=True)

loss = layers.softmax_with_cross_entropy(preds, target_pop_first)
loss = layers.elementwise_mul(loss, target_mask_pop_first)
loss = layers.reduce_sum(loss)

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(loss)    

train_program = fluid.default_main_program().clone()
startup_program = fluid.default_startup_program()

place = fluid.CUDAPlace(1)
exe = fluid.Executor(place)
exe.run(startup_program)

# Inference / translation functions ---------------------------------------
# they are appearing here already in the file because we want to watch how
# the network learns

def plot_attention(attention, sentence, predicted_sentence):
    fig = plt.figure(figsize=(3, 3))
    ax = fig.add_subplot(1, 1, 1)
    ax.matshow(attention, cmap='viridis')

    fontdict = {'fontsize': 7}

    ax.set_xticklabels([''] + sentence, fontdict=fontdict)
    ax.set_yticklabels([''] + predicted_sentence, fontdict=fontdict)

    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    plt.show()

def translate(sentence):
    tar_sentence_ids, attention_mat = inference(sentence)
    if tar_sentence_ids is not None:
        tar_sentence = []
        for i in range(len(tar_sentence_ids)):
            tar_sentence.append(tar_idx_dict[tar_sentence_ids[i]])

        def remove_tags(sentence):
            ret = ""
            for word in sentence:
                if word.find('<') < 0:
                    ret += word + ' '
            return ret

        print("%s -> %s" % (sentence, remove_tags(tar_sentence)))

        attention_mat = attention_mat[0, :, :, 0]
        src_sentence = preprocess_sentence(sentence).split(' ')
        attention_plot = attention_mat[:len(tar_sentence), :len(src_sentence)]
        # plot_attention(attention_plot, src_sentence, tar_sentence)

def inference(sentence):

    fluid.io.load_params(exe, './trained')

    target_batch_accum = \
        np.array([[[tar_word_dict['<start>']]]] * 1).astype('int64')

    input_texts = [preprocess_sentence(sentence)]
    input_diglist = sentlist2diglist(input_texts, src_word_dict)
    input_matrix, _ = pad_sequences(input_diglist, src_maxlen)
    input_batch = np.array(input_matrix).astype('int64')
    input_batch = np.expand_dims(input_batch, axis=-1)

    for i in range(tar_maxlen - 1):

        batch_output = exe.run(
            test_program,
            feed={'input' : input_batch,
                  'target_pop_last' : target_batch_accum},
            fetch_list=[preds_out.name, attention_matrix.name])

        atte_mat = batch_output[1]
        pred_idx = np.argmax(batch_output[0], axis=-1)[:, -1]
        pred_idx = np.expand_dims(pred_idx, axis=-1)
        pred_idx = np.expand_dims(pred_idx, axis=-1)
        target_batch_accum = np.concatenate([target_batch_accum, pred_idx], axis=1)

        pred_word = tar_idx_dict[int(pred_idx[0, -1, 0])]
        if pred_word == '<stop>':
            return np.reshape(target_batch_accum[0], (-1, )), atte_mat

# Training loop -----------------------------------------------------------

def train():
    # fluid.io.load_params(exe, './trained')

    n_epochs = 50

    for epoch in range(n_epochs):

        print('\n epoch: %d' % (epoch))
        num_batches = int(np.ceil(split * samples / float(batch_size)))
        progress_bar = Bar('', max=num_batches)

        total_loss = 0
        for input_batch, target_batch, target_mask_batch in train_dataset():

            target_batch_pop_last = target_batch[:, :tar_maxlen - 1, :]
            target_batch_pop_first = target_batch[:, 1:, :]
            target_batch_mask_pop_first = target_mask_batch[:, 1:, :]

            batch_output = exe.run(
                train_program,
                feed={'input' : input_batch,
                      'target_pop_last' : target_batch_pop_last,
                      'target_pop_first' : target_batch_pop_first,
                      'target_mask_pop_first' : target_batch_mask_pop_first},
                fetch_list=[loss.name])
            total_loss += batch_output[0] / target_batch.shape[1]
            progress_bar.next()
        progress_bar.finish()

        print('total_loss: %f' % (total_loss / batch_size))

    shutil.rmtree('./trained', ignore_errors=True)
    os.makedirs('./trained')
    fluid.io.save_params(executor=exe, dirname='./trained')

if __name__ == '__main__':
    train()
    translate("Hug me .")