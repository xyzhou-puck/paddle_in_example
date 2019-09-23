from __future__ import absolute_import, division, print_function, unicode_literals
import re
import numpy as np
import os, shutil
import time
import json
import cv2
from glob import glob
import pickle
import random
import string
import numpy as np
import paddle
import paddle.fluid as fluid
import matplotlib.pyplot as plt
from PIL import Image
from paddle.fluid.optimizer import AdamOptimizer
from paddle.fluid.dygraph.nn import Conv2D, Pool2D, FC, BatchNorm, Embedding, GRUUnit
from paddle.fluid.dygraph.base import to_variable
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle
import inception_v4
from tqdm import tqdm

annotation_file = "./annotations/captions_train2014.json"
image_path = "./train2014"

print('load annotations...')
with open(annotation_file, 'r') as f:
    annotations = json.load(f)

all_captions = []
all_img_name_vector = []

for annot in annotations['annotations']:
    caption = '<start> ' + annot['caption'] + ' <end>'
    image_id = annot['image_id']
    full_coco_image_path = image_path + '/COCO_train2014_' + '%012d.jpg' % (image_id)

    all_img_name_vector.append(full_coco_image_path)
    all_captions.append(caption)

train_captions, img_name_vector = shuffle(all_captions, 
                                        all_img_name_vector, 
                                        random_state=1)

num_examples = 30000
train_captions = train_captions[:num_examples]
img_name_vector = img_name_vector[:num_examples]

def load_image(image_path):
    tar_size = inception_v4.train_parameters['input_size'][1]
    img = cv2.imread(image_path)
    img = cv2.resize(img, (tar_size, tar_size))
    img = np.array(img).astype('float32').transpose((2, 0, 1))
    img = img / 255
    img -= np.array(inception_v4.train_parameters['input_mean']).reshape((3, 1, 1))
    img /= np.array(inception_v4.train_parameters['input_std']).reshape((3, 1, 1))
    img = np.expand_dims(img, axis=0)
    return img

def batch_creator(path_data, batch_size, drop_last=False):
    def batch_generator():
        batch_img, batch_path = [], []
        data = zip(path_data)
        for path in data:
            img = load_image(path[0])
            batch_img.append(img[0])
            batch_path.append(path)
            if len(batch_img) >= batch_size:
                yield np.array(batch_img), np.array(batch_path)
                batch_img, batch_path = [], []
        if batch_img and not drop_last:
            yield np.array(batch_img), np.array(batch_path)
    return batch_generator

new_input = fluid.layers.data(name='input', shape=[-1, 3, 224, 224], dtype='float32')
image_model = inception_v4.InceptionV4()
hidden_layer = image_model.net(new_input, include_top=False)
image_feature_extract_program = fluid.default_main_program().clone(for_test=True)

encode_train = sorted(set(img_name_vector))
image_dataset = batch_creator(encode_train, 32)

place = fluid.CPUPlace()
exe = fluid.Executor(place)
exe.run(fluid.default_startup_program())

# ~13 min on GTX 1080Ti


fluid.io.load_params(exe, './pretrained', image_feature_extract_program)

print('extracting features...')
num_batches = int(np.ceil(len(encode_train) / 32.0))
for img, path in tqdm(image_dataset(), total=num_batches):
    batch_features = exe.run(
        program=image_feature_extract_program,
        feed={'input' : img},
        fetch_list=[hidden_layer.name])[0]
    batch_features = np.reshape(batch_features, (-1, 1536, 25))

    for bf, p in zip(batch_features, path):
        path_of_feature = p[0]
        np.save(path_of_feature, bf)


# Find the maximum length of any caption in our dataset
def calc_max_length(tensor):
    return max(len(t) for t in tensor)

# Choose the top 5000 words from the vocabulary
def text_tokenizer(text, max_words, reserve):
    freq = dict()
    punc = string.punctuation
    punc = punc.replace('<', '').replace('>', '')
    words = text.translate(str.maketrans('', '', punc))
    words = words.replace('\n', ' ')
    words = words.split(' ')

    for word in words:
        if isinstance(word, str):
            word_low = word.lower()
            if word_low in freq:
                freq[word_low] += 1
            else:
                freq[word_low] = 1

    freq_sorted = sorted(freq.items(), key=lambda kv:(kv[1], kv[0]), reverse=True)

    word2idx, idx2word = dict(), dict()
    count = reserve
    for key, value in freq_sorted: 
        if count < max_words:
            word2idx[key] = count
            idx2word[count] = key
            count += 1

    return word2idx, idx2word

def text_to_sequence(text, w2i, oov_token):
    punc = string.punctuation
    punc = punc.replace('<', '').replace('>', '')
    words = text.translate(str.maketrans('', '', punc))
    words = words.replace('\n', ' ')
    words = words.split(' ')
    text_ids = []
    for word in words:
        if isinstance(word, str):
            word_low = word.lower()
            if word_low in w2i:
                text_ids.append(int(w2i[word_low]))
            else:
                text_ids.append(w2i[oov_token])
    return text_ids

def texts_to_sequences(texts, w2i, oov_token):
    texts_ids = []
    for text in texts:
        texts_ids.append(text_to_sequence(text, w2i, oov_token))
    return texts_ids

def pad_sequences(train_seqs, max_len, pad_id=0):
    train_seqs_w_padding = []
    for seq in train_seqs:
        if len(seq) <= max_len:
            train_seqs_w_padding.append(seq + ([pad_id] * (max_len - len(seq))))
        else:
            train_seqs_w_padding.append(seq[:max_len])
    return train_seqs_w_padding

def ids_to_texts(ids, i2w):
    texts = ""
    for id in ids:
        if id != 0 and id != 1 and id != w2i['<start>'] and id != w2i['<end>']:
            texts += i2w[id] + ' '
    return texts

print('tokenizing...')

top_k = 5000
w2i, i2w = text_tokenizer(' '.join(train_captions), top_k, 2)
w2i['<pad>'] = 0
w2i['<unk>'] = 1
i2w[0] = '<pad>'
i2w[1] = '<unk>'
train_seqs = texts_to_sequences(train_captions, w2i, '<unk>')

max_len = calc_max_length(train_seqs)

cap_vector = pad_sequences(train_seqs, max_len)

# Create training and validation sets using an 80-20 split
img_name_train, img_name_val, cap_train, cap_val = train_test_split(img_name_vector,
                                                                    cap_vector,
                                                                    test_size=0.2,
                                                                    random_state=0)

print('max_len: ', max_len)
print(("%d, %d, %d, %d") % (len(img_name_train), 
                            len(cap_train), 
                            len(img_name_val), 
                            len(cap_val)))

# Feel free to change these parameters according to your system's configuration

BATCH_SIZE = 64
BUFFER_SIZE = 1000
embedding_dim = 256
units = 512
vocab_size = len(w2i)
num_steps = len(img_name_train) // BATCH_SIZE
# Shape of the vector extracted from InceptionV4 is (25, 1536)
# These two variables represent that vector shape
features_shape = 1536
attention_features_shape = 25

def batch_creator(path_data, cap_data, batch_size, drop_last=True):
    def batch_generator():
        batch_img, batch_cap = [], []
        data = zip(path_data, cap_data)
        for path, cap in data:
            img = np.load(path + '.npy')
            batch_img.append(img)
            batch_cap.append(cap)
            if len(batch_img) >= batch_size:
                yield np.array(batch_img), np.array(batch_cap).astype('int64')
                batch_img, batch_cap = [], []
        if batch_img and not drop_last:
            yield np.array(batch_img), np.array(batch_cap).astype('int64')
    return batch_generator

# Load the numpy files
dataset = batch_creator(img_name_train, cap_train, BATCH_SIZE)

class BahdanauAttention(fluid.dygraph.Layer):
    def __init__(self, name_scope, units):
        super(BahdanauAttention, self).__init__(name_scope)

        self.W1 = FC(self.full_name(), size=units, num_flatten_dims=2)
        self.W2 = FC(self.full_name(), size=units, num_flatten_dims=2)
        self.V = FC(self.full_name(), size=1, num_flatten_dims=2)

    def forward(self, features, hidden):
        hidden_with_time_axis = fluid.layers.reshape(hidden, (-1, 1, hidden.shape[-1]))
        hidden_with_time_axis = fluid.layers.expand(hidden_with_time_axis, (1, features.shape[1], 1))
        score = fluid.layers.tanh(self.W1(features) + self.W2(hidden_with_time_axis))
        attention_weights = fluid.layers.softmax(self.V(score), axis=1) # (batch x 25 x 1)

        context_vector = fluid.layers.elementwise_mul(features, attention_weights, axis=0)
        context_vector = fluid.layers.reduce_sum(context_vector, dim=1) # (batch x 256)

        return context_vector, attention_weights

class DynamicGRU(fluid.dygraph.Layer):
    def __init__(self, name_scope, size):
        super(DynamicGRU, self).__init__(name_scope)

        self.gru_unit = GRUUnit(self.full_name(), size * 3)

    def forward(self, inputs, h_0=None):
        hidden = h_0
        res = []

        # input == (batch x seq x hidden)
        for i in range(inputs.shape[1]):
            input_ = inputs[:, i:i + 1, :]
            input_ = fluid.layers.reshape(input_, [-1, input_.shape[2]], inplace=False)
            hidden, reset, gate = self.gru_unit(input_, hidden)
            hidden_ = fluid.layers.reshape(hidden, [-1, 1, hidden.shape[1]], inplace=False)
            res.append(hidden_)

        res = fluid.layers.concat(res, axis=-1)
        return res, hidden

class CNNEncoder(fluid.dygraph.Layer):
    def __init__(self, name_scope, embedding_dim):
        super(CNNEncoder, self).__init__(name_scope)

        self.fc = FC(self.full_name(), size=embedding_dim, num_flatten_dims=2)

    def forward(self, x):
        x = self.fc(x)
        x = fluid.layers.relu(x)
        return x

class RNNDecoder(fluid.dygraph.Layer):
    def __init__(self, name_scope, embedding_dim, units, vocab_size):
        super(RNNDecoder, self).__init__(name_scope)

        self.units = units

        self.embedding = Embedding(self.full_name(), (vocab_size, embedding_dim))
        self.gru = DynamicGRU(self.full_name(), units)
        self.fc1 = FC(self.full_name(), size=units, num_flatten_dims=2)
        self.fc2 = FC(self.full_name(), vocab_size)
        self.attention = BahdanauAttention(self.full_name(), units)

    def forward(self, x, features, hidden):
        context_vector, attention_weights = self.attention(features, hidden)
        context_vector = fluid.layers.reshape(context_vector, (-1, 1, embedding_dim))
        x = self.embedding(x)
        x = fluid.layers.concat([context_vector, x], axis=-1)
        x = fluid.layers.expand(x, (1, 1, 3))
        output, state = self.gru(x, hidden)
        x = self.fc1(output)
        x = fluid.layers.reshape(x, (-1, x.shape[2]))
        x = self.fc2(x)
        return x, state, attention_weights

# segfault on GPU
def cx_loss(y_true, y_pred):
    zeros = fluid.layers.zeros_like(y_true)
    mask  = fluid.layers.logical_not(fluid.layers.equal(y_true, zeros))
    mask  = fluid.layers.cast(mask, dtype='float32')
    mask.stop_gradient=True
    loss_ = fluid.layers.softmax_with_cross_entropy(y_pred, y_true) * mask
    return fluid.layers.reduce_sum(loss_)

def cx_loss_np(y_true, y_pred):
    zeros_np = np.zeros(y_true.shape)
    mask_np = np.logical_not(np.equal(y_true.numpy(), zeros_np)).astype('float32')
    mask = to_variable(mask_np)
    loss_ = fluid.layers.softmax_with_cross_entropy(y_pred, y_true) * mask
    return fluid.layers.reduce_sum(loss_)

# ~2.5min per epoch on GTX 1080

def train():
    place = fluid.CUDAPlace(1)
    with fluid.dygraph.guard(place):

        encoder = CNNEncoder('encoder', embedding_dim)
        decoder = RNNDecoder('decoder', embedding_dim, units, vocab_size)
        encoder_weight, _ = fluid.dygraph.load_persistables('./weight/encoder/')
        decoder_weight, _ = fluid.dygraph.load_persistables('./weight/decoder/') 
        encoder.load_dict(encoder_weight)
        decoder.load_dict(decoder_weight)

        optimizer = fluid.optimizer.Adam(learning_rate=1e-3)

        EPOCHS = 20

        for epoch in range(EPOCHS):
            total_loss = 0
            print('epoch: ', epoch + 1)
            for img, cap in tqdm(dataset(), total=num_steps):
                batch_loss = 0
                hidden = to_variable(np.zeros((BATCH_SIZE, units)).astype('float32'))
                dec_input = to_variable(np.expand_dims(np.array([[w2i['<start>']]] * BATCH_SIZE), -1))
                img_tensor = to_variable(img)
                cap_tensor = to_variable(np.expand_dims(cap, axis=-1))

                # (batch x embedding_dim x 25) -> (batch x 25 x embedding_dim)
                img_tensor = fluid.layers.transpose(img_tensor, (0, 2, 1))

                features = encoder(img_tensor)

                for i in range(1, cap_tensor.shape[1]):
                    # passing the features through the decoder
                    predictions, hidden, _ = decoder(dec_input, features, hidden)

                    loss = cx_loss_np(cap_tensor[:, i], predictions)

                    batch_loss += loss

                    dec_input = to_variable(np.expand_dims(np.expand_dims(cap, axis=-1)[:, i], axis=-1))
                    
                total_loss += batch_loss / float(cap_tensor.shape[1])

                batch_loss.backward()
                optimizer.minimize(loss)
                encoder.clear_gradients()
                decoder.clear_gradients()
                
            print('total_loss: ', (total_loss.numpy()[0] / float(num_steps)))   
            fluid.dygraph.save_persistables(encoder.state_dict(), './weight/encoder/')
            fluid.dygraph.save_persistables(decoder.state_dict(), './weight/decoder/')                                                     

def get_caption(image):

    fluid.io.load_params(exe, './pretrained', image_feature_extract_program)

    temp_input = load_image(image)
    img_tensor_val = exe.run(
        program=image_feature_extract_program,
        feed={'input' : temp_input},
        fetch_list=[hidden_layer.name])[0]
    img_tensor_val = np.reshape(img_tensor_val, (-1, 1536, 25))

    place = fluid.CPUPlace()
    with fluid.dygraph.guard(place):

        encoder = CNNEncoder('encoder', embedding_dim)
        decoder = RNNDecoder('decoder', embedding_dim, units, vocab_size)
        encoder_weight, _ = fluid.dygraph.load_persistables('./weight/encoder/')
        decoder_weight, _ = fluid.dygraph.load_persistables('./weight/decoder/') 
        encoder.load_dict(encoder_weight)
        decoder.load_dict(decoder_weight)

        attention_plot = np.zeros((max_len, attention_features_shape))
        hidden = to_variable(np.zeros((1, units)).astype('float32'))
        img_tensor = to_variable(img_tensor_val)

        img_tensor = fluid.layers.transpose(img_tensor, (0, 2, 1))
        
        features = encoder(img_tensor)

        dec_input = to_variable(np.expand_dims(np.array([[w2i['<start>']]]), -1))
        result = []

        for i in range(max_len):

            predictions, hidden, attention_weights = decoder(dec_input, features, hidden)

            attention_plot[i] = fluid.layers.reshape(attention_weights, (-1, )).numpy()

            predicted_id = np.argmax(predictions[0].numpy())

            result.append(i2w[predicted_id])

            if i2w[predicted_id] == '<end>':
                result = result[:len(result) - 1]
                attention_plot = attention_plot[:len(result), :]
                return result, attention_plot

            dec_input = to_variable(np.expand_dims([[predicted_id]], axis=-1))
        
        attention_plot = attention_plot[:len(result - 1), :]
        return result, attention_plot 

def plot_attention(image, result, attention_plot):
    temp_image = np.array(Image.open(image))

    fig = plt.figure(figsize=(7, 7))

    len_result = len(result)
    for l in range(len_result):
        temp_att = np.resize(attention_plot[l], (5, 5))
        ax = fig.add_subplot(len_result//2, len_result//2, l+1)
        ax.set_title(result[l])
        img = ax.imshow(temp_image)
        ax.imshow(temp_att, cmap='gray', alpha=0.6, extent=img.get_extent())

    plt.tight_layout()
    plt.show()

train_images_4checking = [img_name_train[4],  img_name_train[10], img_name_train[30]]
train_captions_4checking = [cap_train[4], cap_train[10], cap_train[30]]
validation_images_4checking = [img_name_val[7], img_name_val[10], img_name_val[12]]
validation_captions_4checking = [cap_val[7], cap_val[10], cap_val[12]]

def check_sample_captions(mode='training', plot=False):
    if mode == 'training':
        images = train_images_4checking
        captions = train_captions_4checking
    elif mode == 'validation':
        images = validation_images_4checking
        captions = validation_captions_4checking
    print('\nSample checks on', mode)

    for i in range(len(images)):
        result, attention_plot = get_caption(images[i])
        print('\nReal Caption:', ids_to_texts(captions[i], i2w))
        print('Prediction Caption:', ' '.join(result))
        if plot:
            plot_attention(images[i], result, attention_plot)

if __name__ == '__main__':
    # train()
    check_sample_captions(mode='training', plot=True)
    check_sample_captions(mode='validation', plot=True)
