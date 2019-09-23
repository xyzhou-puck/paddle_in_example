import os, shutil
import wget, zipfile
import cv2
import warnings
import random
import string
import csv
import re
import numpy as np
import paddle
import paddle.fluid as fluid
from lime.lime_text import LimeTextExplainer
from collections import OrderedDict 
import matplotlib.pyplot as plt

# Download and unzip data

print('downloading...')
url =  "https://archive.ics.uci.edu/ml/machine-learning-databases/00461/drugLib_raw.zip"
wget.download(url, './drugLibTest_raw.zip')

print('unziping...')
with zipfile.ZipFile('./drugLibTest_raw.zip', 'r') as zip_ref:
    zip_ref.extractall('./data')

# Read dataset

text = [] # This is our text
y_train = [] # And these are ratings given by customers

# Select only rating and text from the whole dataset
with open("./data/drugLibTest_raw.tsv") as df:
    rd = csv.reader(df, delimiter="\t", quotechar='"')
    for i, row in enumerate(rd):
        if i > 0:
            if int(row[2]) >= 8:
                rating = 0
            else:
                rating = 1
            text.append(row[6])
            y_train.append([rating])
y_train = np.array(y_train).astype('int64')

# text_tokenizer helps us to turn each word into integers. By selecting maximum number of features
# we also keep the most frequent words. Additionally, by default, all punctuation is removed.

def text_tokenizer(text, max_words):
    freq = dict()
    remove_punc = text.translate(str.maketrans('', '', string.punctuation))
    words = re.findall(r"[\w']+", remove_punc)
    for word in words:
        if isinstance(word, str):
            word_low = word.lower()
            if word_low in freq:
                freq[word_low] += 1
            else:
                freq[word_low] = 1

    freq_sorted = sorted(freq.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)

    word2idx, idx2word = dict(), dict()
    count = 0
    for key, value in freq_sorted: 
        if count < max_words:
            word2idx[key] = count
            idx2word[count] = key
            count += 1

    return word2idx, idx2word

# Then, we need to fit the tokenizer object to our text data

max_features = 1000
w2i, i2w = text_tokenizer(' '.join(text), max_features)

# Via tokenizer object you can check word indices, word counts and other interesting properties.

# print(w2i)

# Finally, we can replace words in dataset with integers

def text_to_sequence(text, w2i):
    remove_punc = text.translate(str.maketrans('', '', string.punctuation))
    words = re.findall(r"[\w']+", remove_punc)
    text_ids = []
    for word in words:
        if isinstance(word, str):
            word_low = word.lower()
            if word_low in w2i:
                text_ids.append(int(w2i[word_low]))
            else:
                continue
    return text_ids

def texts_to_sequences(texts, w2i):
    texts_ids = []
    for text in texts:
        texts_ids.append(text_to_sequence(text, w2i))
    return texts_ids

text_seqs = texts_to_sequences(text, w2i)

# Define the parameters of the keras model

maxlen = 30
batch_size = 32
embedding_dims = 50
filters = 64
kernel_size = 3
hidden_dims = 50
epochs = 15

# As a final step, restrict the maximum length of all sequences and create a matrix as input for model

def cut(texts, maxlen):
    ret_texts = []
    for text in texts:
        if len(text) > maxlen:
            ret_texts.append(text[:15])
        else:
            ret_texts.append(text)
    return ret_texts

def to_lodtensor(data, place = fluid.CPUPlace()):
    seq_lens = [len(seq) for seq in data]
    cur_len = 0
    lod = [cur_len]
    for l in seq_lens:
        cur_len += l
        lod.append(cur_len)
    flattened_data = np.concatenate(data, axis=0).astype("int64")
    flattened_data = flattened_data.reshape([len(flattened_data), 1])

    res = fluid.LoDTensor()
    res.set(flattened_data, place)
    res.set_lod([lod])
    return res

x_train = cut(text_seqs, maxlen)

# Lets print the first 2 rows and see that max length of first 2 sequences equals to 15

# print(x_train[0:2])

# Create a model

x = fluid.layers.data(name="x", shape=[1], dtype='int64', lod_level=1)
y = fluid.layers.data(name="y", shape=[1], dtype='int64')

h_emb   = fluid.layers.embedding(x, size=[max_features, embedding_dims])
h_drop1 = fluid.layers.dropout(h_emb, dropout_prob=0.2)
h_conv  = fluid.layers.sequence_conv(h_drop1, num_filters=filters, 
    filter_size=kernel_size, act='relu')
h_pool  = fluid.layers.sequence_pool(h_conv, pool_type='max')
h_fc1   = fluid.layers.fc(h_pool, hidden_dims)
h_drop2 = fluid.layers.dropout(h_fc1, dropout_prob=0.2)
h_act1  = fluid.layers.relu(h_drop2)
pred    = fluid.layers.fc(h_act1, 2)

pred_out = fluid.layers.softmax(pred)

test_program = fluid.default_main_program().clone(for_test=True)

cost = fluid.layers.softmax_with_cross_entropy(pred, y)
loss = fluid.layers.reduce_mean(cost)
acc  = fluid.layers.accuracy(pred, y)

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(loss)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# define data reader
def batch_creator(x_data, y_data, batch_size, drop_last=False):
    def batch_generator():
        batch_x, batch_y = [], []
        data = zip(x_data, y_data)
        for x, y in data:
            batch_x.append(x)
            batch_y.append(y)
            if len(batch_x) >= batch_size:
                yield to_lodtensor(batch_x), np.array(batch_y)
                batch_x, batch_y = [], []
        if batch_x and not drop_last:
            yield to_lodtensor(batch_x), np.array(batch_y)
    return batch_generator

train_dataset = batch_creator(x_train, y_train, batch_size)

for epoch in range(epochs):
    print("epoch ", epoch)
    mean_loss, mean_acc = [], []
    for batch_x, batch_y in train_dataset():
        out = exe.run(
            feed={'x' : batch_x, 'y' : batch_y}, 
            fetch_list=[loss.name, acc.name])
        mean_loss.append(float(out[0]))
        mean_acc.append(float(out[1]))
    mean_loss = np.array(mean_loss).mean()
    mean_acc = np.array(mean_acc).mean()
    print("train loss: %.6f, acc: %.4f" % (mean_loss, mean_acc))

# Understanding lime for Keras Embedding Layers

# In order to explain a text with LIME, we should write a preprocess function
# which will help to turn words into integers. Therefore, above mentioned steps 
# (how to encode a text) should be repeated BUT within a function. 
# As we already have had a tokenizer object, we can apply the same object to train/test or a new text.

def inference(x_batch):
    return exe.run(
        program = test_program,
        feed={'x' : to_lodtensor(x_batch)}, 
        fetch_list=[pred_out.name])[0]

def get_embedding_explanation(input_text, w2i):
    text_to_seq = texts_to_sequences(input_text, w2i)
    cut_seq = cut(text_to_seq, maxlen)
    return cut_seq

def classifer(input_text):
    data = get_embedding_explanation(input_text, w2i)
    pred = inference(data)
    return pred

# Lets choose some text to explain
sentence_to_explain = text[1]
print('\n' + sentence_to_explain + '\n')

explainer = LimeTextExplainer(class_names=['high rating', 'low rating'])
explanation = explainer.explain_instance(
    sentence_to_explain, 
    classifer,
    num_features=10,
    num_samples=10000)

explanation.show_in_notebook(text=True)
explanation.save_to_file('./explanation.html')
