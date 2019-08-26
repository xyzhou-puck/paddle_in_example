# coding=utf8
import numpy as np
from paddle import fluid
from paddle.fluid import layers
import csv
import nltk
from nltk.corpus import reuters
from nltk.corpus import stopwords
from sklearn.preprocessing import MultiLabelBinarizer
from sklearn.feature_extraction.text import TfidfVectorizer


def load_data(config={}):

    stop_words = stopwords.words("english")
    vectorizer = TfidfVectorizer(stop_words=stop_words, binary=True)
    mlb = MultiLabelBinarizer()

    documents = reuters.fileids()
    test = [d for d in documents if d.startswith('test/')]
    train = [d for d in documents if d.startswith('training/')]

    docs = {}
    docs['train'] = [reuters.raw(doc_id) for doc_id in train]
    docs['test'] = [reuters.raw(doc_id) for doc_id in test]
    xs = {'train': [], 'test': []}
    xs['train'] = vectorizer.fit_transform(docs['train']).toarray()
    xs['test'] = vectorizer.transform(docs['test']).toarray()
    ys = {'train': [], 'test': []}
    ys['train'] = mlb.fit_transform([reuters.categories(doc_id)
                                     for doc_id in train])
    ys['test'] = mlb.transform([reuters.categories(doc_id)
                                for doc_id in test])
    data = {'x_train': xs['train'], 'y_train': ys['train'],
            'x_test': xs['test'], 'y_test': ys['test'],
            'labels': reuters.categories()}
    print(data['x_train'])
    print(data['y_train'])
    return data, vectorizer.vocabulary_


def build_data_layer(vocab_size, n_classes, sync=True, inputs_generator_fn=None):

    x = layers.data('x', shape=[-1, vocab_size], dtype='float32')
    y = layers.data('y', shape=[-1, n_classes], dtype='float32')

    ret = [x, y]

    if not sync:
        assert inputs_generator_fn is not None
        reader = fluid.io.PyReader(ret, capacity=1, iterable=False)
        reader.decorate_batch_generator(inputs_generator_fn)
        reader.start()

    return ret


def create_batch_generator(examples, labels, batch_size, num_epochs=1, is_train=False):
    
    def batch_generator_fn():
        batch_x = []
        batch_y = []
        for i in range(num_epochs):
            if is_train:
                print('Training epoch {}:'.format(i))
            for _x, _y in zip(examples, labels):
                batch_x.append(_x)
                batch_y.append(_y)
                if len(batch_x) == batch_size:
                    batch_x = array_normalize(batch_x, dtype='float32')
                    batch_y = array_normalize(batch_y, dtype='float32')
                    yield (batch_x, batch_y)
                    batch_x = []
                    batch_y = []

    return batch_generator_fn


def array_normalize(x, dtype=None):
    x = np.array(x)
    if dtype is not None:
        x = x.astype(dtype)
    return x


def mlp(x, conf):
    hid = layers.fc(x, conf['hidden_size'])
    hid = layers.dropout(hid, conf['dropout_prob'])
    pred = layers.fc(hid, conf['n_classes'])
    return pred


def build_train_program(conf):
    
    train_prog = fluid.Program()
    init_prog = fluid.Program()
    with fluid.program_guard(train_prog, init_prog):
        x, y = build_data_layer(conf['vocab_size'], conf['n_classes'])
        prediction = mlp(x, conf)
        loss = fluid.layers.sigmoid_cross_entropy_with_logits(prediction, label=y)
        loss = fluid.layers.reduce_mean(loss)
        adam_optimizer = fluid.optimizer.Adam(learning_rate=0.001)
        adam_optimizer.minimize(loss)

    fetch_list = [loss]
    return init_prog, train_prog, fetch_list, [x.name, y.name]


def build_test_program(conf):
    
    prog = fluid.Program()
    with fluid.program_guard(prog):
        x, y = build_data_layer(conf['vocab_size'], conf['n_classes'])
        prediction = mlp(x, conf)
        prediction = layers.sigmoid(prediction)
        prediction = prediction > 0.5

    fetch_list = [prediction]
    return prog, fetch_list, [x.name]


def build_executor(use_gpu=False):

    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    return fluid.Executor(place)


if __name__ == '__main__':

    # labels = reuters.categories()
    conf = {
        "n_classes": 90,
        "batch_size": 64,
        "num_epochs": 20,
        "hidden_size": 256,
        "dropout_prob": 0.5,
    }

    print "prepara data..."
    data, vocab = load_data()
    conf["vocab_size"] = len(vocab)

    # build batch generator
    train_generator_fn = create_batch_generator(data['x_train'], data['y_train'], conf['batch_size'], conf['num_epochs'], is_train=True)

    # build train and eval program
    init_prog, train_prog, fetch_list, feed_list = build_train_program(conf)
    with fluid.unique_name.guard():    
        test_prog, test_fetch_list, test_feed_list = build_test_program(conf)

    # build executor
    exe = build_executor(use_gpu=True)

    # initialize
    exe.run(init_prog)

    # do train
    print "training..."
    steps = 0
    for batch in train_generator_fn():
        steps += 1
        loss = exe.run(train_prog, fetch_list=fetch_list, feed = {i: j for i,j in zip(feed_list, batch)})[0]
        if steps % 20 == 0:
            print("step {}, loss {}.".format(steps, loss))

    # do eval
    print "evaling..."
    eval_generator_fn = create_batch_generator(data['x_test'], data['y_test'], conf['batch_size'], is_train=False)

    correct = 0
    cnt = 0
    for x,y in eval_generator_fn():
        pred = exe.run(test_prog, fetch_list=test_fetch_list, feed = {test_feed_list[0]: x})[0]
        
        for i,j in zip(pred, y):
            if i.tolist() == j.tolist():
                correct += 1
            cnt += 1
    print "acc:"
    print correct / float(cnt)

