# coding=utf8
import nltk
from collections import Counter
import numpy as np
from paddle import fluid
from paddle.fluid import layers
import csv


def build_data_layer(max_len, sync=True, inputs_generator_fn=None):

    x1 = layers.data('tok1', shape=[-1, max_len, 1], dtype='int64')
    x2 = layers.data('tok2', shape=[-1, max_len, 1], dtype='int64')
    l1 = layers.data('len1', shape=[-1], dtype='int64')
    l2 = layers.data('len2', shape=[-1], dtype='int64')
    y = layers.data('y', shape=[-1, 1], dtype='int64')

    if not sync:
        assert inputs_generator_fn is not None
        reader = fluid.io.PyReader([x1, x2, l1, l2, y], capacity=1, iterable=False)
        reader.decorate_batch_generator(inputs_generator_fn)
        reader.start()

    return x1, x2, l1, l2, y


def create_batch_generator(examples, labels, batch_size, max_len, num_epochs=1, is_train=False):
    
    def batch_generator_fn():
        batch_x1 = []
        batch_x2 = []
        batch_l1 = []
        batch_l2 = []
        batch_y = []
        for i in range(num_epochs):
            print('Training epoch {}:'.format(i))
            for (_x1, _x2), _y in zip(examples, labels):
                _x1 = _x1[:max_len]
                _x2 = _x2[:max_len]
                batch_x1.append(_x1)
                batch_x2.append(_x2)
                batch_l1.append(len(_x1))
                batch_l2.append(len(_x2))
                batch_y.append(_y)
                if len(batch_x1) == batch_size:
                    batch_x1 = array_normalize(batch_x1, pad_to_len=max_len)
                    batch_x2 = array_normalize(batch_x2, pad_to_len=max_len)
                    batch_l1 = array_normalize(batch_l1, expand_dims=False)
                    batch_l2 = array_normalize(batch_l2, expand_dims=False)
                    batch_y = array_normalize(batch_y)

                    yield (batch_x1, batch_x2, batch_l1, batch_l2, batch_y)
                    batch_x1 = []
                    batch_x2 = []
                    batch_l1 = []
                    batch_l2 = []
                    batch_y = []

    return batch_generator_fn


def array_normalize(x, dtype=None, expand_dims=True, pad_to_len=None):
    if pad_to_len is not None:
        x = [i + [0] * (pad_to_len-len(i)) for i in x]

    x = np.array(x)
    if dtype is not None:
        x = x.astype(dtype)
    if expand_dims:
        x = np.expand_dims(x, -1)
    return x


def siamLSTM(tok_ids1, tok_ids2, len1, len2, conf):

    emb = fluid.ParamAttr('embedding', initializer=fluid.initializer.UniformInitializer(-0.1, 0.1))
    emb1 = layers.embedding(tok_ids1, size=[conf['vocab_size'], conf['hidden_size']], dtype='float32', is_sparse=False, param_attr=emb)
    emb2 = layers.embedding(tok_ids2, size=[conf['vocab_size'], conf['hidden_size']], dtype='float32', is_sparse=False, param_attr=emb)

    tmp = fluid.ParamAttr('lstm')
    _, enc_out1, _ = fluid.contrib.layers.basic_lstm(emb1, None, None, conf['hidden_size'], sequence_length=len1, param_attr=tmp)
    _, enc_out2, _ = fluid.contrib.layers.basic_lstm(emb2, None, None, conf['hidden_size'], sequence_length=len2, param_attr=tmp)

    enc_out1 = layers.squeeze(enc_out1, [0])
    enc_out2 = layers.squeeze(enc_out2, [0])

    sim = layers.fc(enc_out1 * enc_out2, 2)
    
    return sim


def build_train_program(conf, data_gen_fn):
    
    train_prog = fluid.Program()
    init_prog = fluid.Program()
    with fluid.program_guard(train_prog, init_prog):
        tok1, tok2, len1, len2, y = build_data_layer(conf['max_len'])
        prediction = siamLSTM(tok1, tok2, len1, len2, conf)
        loss = fluid.layers.softmax_with_cross_entropy(prediction, label=y)
        loss = fluid.layers.reduce_mean(loss)
        accuracy = fluid.layers.accuracy(input=prediction, label=y)
        adam_optimizer = fluid.optimizer.Adam(learning_rate=0.005)
        adam_optimizer.minimize(loss)

    fetch_list = [loss, accuracy]
    return init_prog, train_prog, fetch_list, [tok1.name, tok2.name, len1.name, len2.name, y.name]


def build_test_program(conf):
    
    prog = fluid.Program()
    with fluid.program_guard(prog):
        tok1, tok2, len1, len2, y = build_data_layer(conf['max_len'])
        prediction = siamLSTM(tok1, tok2, len1, len2, conf)
        prediction = layers.softmax(prediction)

    fetch_list = [prediction]
    return prog, fetch_list, [tok1.name, tok2.name, len1.name, len2.name]


def build_executor(use_gpu=False):

    if use_gpu:
        place = fluid.CUDAPlace(0)
    else:
        place = fluid.CPUPlace()
    return fluid.Executor(place)


if __name__ == '__main__':

    conf = {
        "batch_size": 64,
        "vocab_size": 50000,
        "num_epochs": 2,
        "max_len": 30,
        "hidden_size": 256,
    }

    print "read raw data and shuffle..."
    reader = csv.reader(open('./quora_duplicate_questions.tsv'), delimiter='\t')
    raw_data = [i[3:6] for i in reader]
    del raw_data[0]
    np.random.shuffle(raw_data)

    # build dict
    print "build dict..."
    tokens = []
    labels = []
    word_freq = Counter()
    for x1, x2, y in raw_data:
        labels.append(int(y))
        tok1 = nltk.word_tokenize(x1.decode('utf8').lower())
        tok2 = nltk.word_tokenize(x2.decode('utf8').lower())
        word_freq.update(tok1)
        word_freq.update(tok2)
        tokens.append(tok1)
        tokens.append(tok2)

    word_freq = sorted(word_freq.items(), key=lambda i: i[1], reverse=True)
    word_to_id = {w: id for id, (w, _) in enumerate(word_freq[:conf["vocab_size"]], start=2)}
    word_to_id['<PAD>'] = 0
    word_to_id['<UNK>'] = 1
    conf['vocab_size'] = len(word_to_id)

    # convert tokens to ids
    print "processing..."
    examples = [[word_to_id.get(w, 1) for w in t] for t in tokens]
    examples = np.reshape(examples, [-1, 2])

    # build batch generator
    train_generator_fn = create_batch_generator(examples, labels, conf['batch_size'], conf['max_len'], conf['num_epochs'], is_train=True)

    # build train and eval program
    init_prog, train_prog, fetch_list, feed_list = build_train_program(conf, train_generator_fn)
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
        loss, acc = exe.run(train_prog, fetch_list=fetch_list, feed = {i: j for i,j in zip(feed_list, batch)})
        if steps % 20 == 0:
            print("step {}, loss {}, acc {}.".format(steps, loss, acc))

    # eval
    print "evaling..."
    q1 = "What is the benefit of Quora?"
    q2 = "What are the advantages of using Quora?"

    print "input pair: \n{}\n{}\n".format(q1,q2)
    q1 = nltk.word_tokenize(q1.lower())
    q2 = nltk.word_tokenize(q2.lower())

    q1 = [[word_to_id.get(w, 1) for w in q1]]
    q2 = [[word_to_id.get(w, 1) for w in q2]]

    q1 = array_normalize(q1, pad_to_len=conf['max_len'])
    q2 = array_normalize(q2, pad_to_len=conf['max_len'])
    len1 = array_normalize([len(q1)], expand_dims=False)
    len2 = array_normalize([len(q2)], expand_dims=False)

    batch = [q1, q2, len1, len2]
    pred = exe.run(test_prog, fetch_list=test_fetch_list, feed={i: j for i,j in zip(test_feed_list, batch)})[0][0][1]
    print "similarity (0 ~ 1):"
    print pred

