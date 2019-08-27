from __future__ import print_function
from functools import reduce
import re
import tarfile
import os
import numpy as np
import argparse

from data_utils import get_file
import paddle
import paddle.fluid as fluid
from pad_sequences import pad_sequences
from paddle.fluid.contrib.layers import basic_lstm

def tokenize(sent):
    '''Return the tokens of a sentence including punctuation.
    # >>> tokenize('Bob dropped the apple. Where is the apple?')
    ['Bob', 'dropped', 'the', 'apple', '.', 'Where', 'is', 'the', 'apple', '?']
    '''
    return [x.strip() for x in re.split(r'(\w+)?', sent) if x and x.strip()]


def parse_stories(lines, only_supporting=False):
    '''Parse stories provided in the bAbi tasks format

    If only_supporting is true,
    only the sentences that support the answer are kept.
    '''
    data = []
    story = []
    for line in lines:
        line = line.strip()
        nid, line = line.split(' ', 1)
        nid = int(nid)
        if nid == 1:
            story = []
        if '\t' in line:
            q, a, supporting = line.split('\t')
            q = tokenize(q)
            if only_supporting:
                # Only select the related substory
                supporting = map(int, supporting.split())
                substory = [story[i - 1] for i in supporting]
            else:
                # Provide all the substories
                substory = [x for x in story if x]
            data.append((substory, q, a))
            story.append('')
        else:
            sent = tokenize(line)
            story.append(sent)
    return data


def get_stories(fname, only_supporting=False, max_length=None):
    '''Given a file name, read the file, retrieve the stories,
    and then convert the sentences into a single story.

    If max_length is supplied,
    any stories longer than max_length tokens will be discarded.
    '''
    f = open(fname, 'r')
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
            if not max_length or len(flatten(story)) < max_length]
    return data


def vectorize_stories(data, word_idx, story_maxlen, query_maxlen):
    xs = []
    xqs = []
    ys = []
    for story, query, answer in data:
        x = [word_idx[w] for w in story]
        xq = [word_idx[w] for w in query]
        # let's not forget that index 0 is reserved
        y = np.zeros(len(word_idx) + 1)
        y[word_idx[answer]] = 1
        xs.append(x)
        xqs.append(xq)
        ys.append(y)
    return (pad_sequences(xs, maxlen=story_maxlen),
            pad_sequences(xqs, maxlen=query_maxlen), np.array(ys))

EMBED_HIDDEN_SIZE = 50
SENT_HIDDEN_SIZE = 100
QUERY_HIDDEN_SIZE = 100
BATCH_SIZE = 32
EPOCHS = 20
print('RNN / Embed / Sent / Query = {}, {}, {}, {}'.format("lstm",
                                                           EMBED_HIDDEN_SIZE,
                                                           SENT_HIDDEN_SIZE,
                                                           QUERY_HIDDEN_SIZE))

try:
    path, datadir = get_file('babi-tasks-v1-2.tar.gz',
                    origin='https://s3.amazonaws.com/text-datasets/'
                           'babi_tasks_1-20_v1-2.tar.gz')
except:
    print('Error downloading dataset, please download it manually:\n'
          '$ wget http://www.thespermwhale.com/jaseweston/babi/tasks_1-20_v1-2'
          '.tar.gz\n'
          '$ mv tasks_1-20_v1-2.tar.gz ~/.keras/datasets/babi-tasks-v1-2.tar.gz')
    raise

# Default QA1 with 1000 samples
# challenge = 'tasks_1-20_v1-2/en/qa1_single-supporting-fact_{}.txt'
# QA1 with 10,000 samples
# challenge = 'tasks_1-20_v1-2/en-10k/qa1_single-supporting-fact_{}.txt'
# QA2 with 1000 samples
# challenge = 'tasks_1-20_v1-2/en/qa2_two-supporting-facts_{}.txt'
# QA2 with 10,000 samples
challenge = 'tasks_1-20_v1-2/en-10k/qa2_two-supporting-facts_{}.txt'

with tarfile.open(path) as tar:
    train = get_stories(os.path.join(datadir, challenge.format('train')))
    test = get_stories(os.path.join(datadir, challenge.format('test')))

vocab = set()
for story, q, answer in train + test:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

# Reserve 0 for masking via pad_sequences
vocab_size = len(vocab) + 1
word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
story_maxlen = max(map(len, (x for x, _, _ in train + test)))
query_maxlen = max(map(len, (x for _, x, _ in train + test)))

x, xq, y = vectorize_stories(train, word_idx, story_maxlen, query_maxlen)
tx, txq, ty = vectorize_stories(test, word_idx, story_maxlen, query_maxlen)

print('vocab_size = {}'.format(vocab_size))
print('x.shape = {}'.format(x.shape))
print('xq.shape = {}'.format(xq.shape))
print('y.shape = {}'.format(y.shape))
print('story_maxlen, query_maxlen = {}, {}'.format(story_maxlen, query_maxlen))

print('Build model...')

def babirnn_neural_network(sentence, question, answer):

    encoded_sentence_emb = fluid.layers.embedding(input=sentence, size=[vocab_size, EMBED_HIDDEN_SIZE], is_sparse=True)
    _, encoded_sentence, _ = basic_lstm(encoded_sentence_emb, None, None, SENT_HIDDEN_SIZE)
    encoded_question_emb = fluid.layers.embedding(input=question, size=[vocab_size, EMBED_HIDDEN_SIZE], is_sparse=True)
    _, encoded_question, _ = basic_lstm(encoded_question_emb, None, None, QUERY_HIDDEN_SIZE)

    merged = fluid.layers.concat(input=[encoded_sentence[0], encoded_question[0]], axis=-1)
    preds = fluid.layers.fc(input=merged, size=vocab_size, act='softmax')

    # loss
    loss = fluid.layers.cross_entropy(input=preds, label=answer, soft_label=True)
    avg_loss = fluid.layers.mean(loss)
    label = fluid.layers.reshape(fluid.layers.argmax(answer, axis=-1), shape=[-1, 1])
    accuracy = fluid.layers.accuracy(input=preds, label=label)

    return preds, avg_loss, accuracy


def train(args):
    if args.use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    def reader_creater(sentence, question, answer):
        def reader():
            for sent, ques, ans in zip(sentence, question, answer):
                yield sent, ques, ans
        return reader

    if args.enable_ce:
        train_reader = paddle.batch(reader_creater(x, xq, y), batch_size=BATCH_SIZE)
        test_reader = paddle.batch(reader_creater(tx, txq, ty), batch_size=BATCH_SIZE)
        startup_program.random_seed = 90
        main_program.random_seed = 90
    else:
        train_reader = paddle.batch(paddle.reader.shuffle(
            reader_creater(x, xq, y), buf_size=1000), batch_size=BATCH_SIZE)
        test_reader = paddle.batch(reader_creater(tx, txq, ty), batch_size=BATCH_SIZE)

    sentence = fluid.layers.data(name="sentence", shape=[story_maxlen,1], dtype='int64', append_batch_size=True)
    question = fluid.layers.data(name="question", shape=[query_maxlen,1], dtype='int64', append_batch_size=True)
    answer = fluid.layers.data(name="answer", shape=[vocab_size], dtype='float32', append_batch_size=True)

    net_conf = babirnn_neural_network

    preds, avg_loss, accuracy = net_conf(sentence, question, answer)

    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        # preds_set = []
        avg_loss_set = []
        accuracy_set = []
        for test_data in train_test_reader():
            preds_np, avg_loss_np, accuracy_np = exe.run(
                program=train_test_program,
                feed=train_test_feed.feed(test_data),
                fetch_list=[preds, avg_loss, accuracy])
            # preds_set.append(float(preds_np))
            avg_loss_set.append(float(avg_loss_np))
            accuracy_set.append(float(accuracy_np))
        # get test acc and loss
        # preds_mean = np.array(preds_set).mean()
        avg_loss_mean = np.array(avg_loss_set).mean()
        accuracy_mean = np.array(accuracy_set).mean()
        #return preds_mean, avg_loss_mean, accuracy_mean
        return avg_loss_mean, accuracy_mean

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[sentence, question, answer], place=place)
    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(EPOCHS)]

    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(
                main_program,
                feed=feeder.feed(data),
                fetch_list=[preds, avg_loss, accuracy])
            step += 1

        print("Epoch %d, loss %f, acc %f" % (epoch_id,
                                             metrics[1], metrics[2]))
        # test for epoch
        test_metrics = train_test(
            train_test_program=test_program,
            train_test_reader=test_reader,
            train_test_feed=feeder)

        print("===Test with Epoch %d, avg_loss %f, accuracy %f===" %
              (epoch_id, test_metrics[0], test_metrics[1]))
        lists.append((epoch_id, test_metrics[0], test_metrics[1]))


    # find the best pass
    best = sorted(lists, key=lambda list: float(list[1]))[0]
    print('Best epoch is %s, val_loss is %s, val_acc is %s' % (best[0], best[1], best[2]))

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("-c", "--enable_ce",
                        help="Whether to enable ce",
                        action='store_true')
    parser.add_argument("-g", "--use_cuda",
                        help="Whether to use GPU to train",
                        default=True)
    args = parser.parse_args()
    train(args)

if __name__ == '__main__':
    main()
