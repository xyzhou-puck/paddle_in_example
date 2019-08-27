from __future__ import print_function
import paddle
import paddle.fluid as fluid
import numpy as np
from six.moves import range
import argparse
from paddle.fluid.contrib.layers import basic_lstm


class CharacterTable(object):

    def __init__(self, chars):

        self.chars = sorted(set(chars))
        self.char_indices = dict((c, i) for i, c in enumerate(self.chars))
        self.indices_char = dict((i, c) for i, c in enumerate(self.chars))

    def encode(self, C, num_rows):

        x = np.zeros((num_rows, len(self.chars)))
        for i, c in enumerate(C):
            x[i, self.char_indices[c]] = 1
        return x

    def decode(self, x, calc_argmax=True):

        if calc_argmax:
            x = x.argmax(axis=-1)
        return ''.join(self.indices_char[x] for x in x)

class colors:
    ok = '\033[92m'
    fail = '\033[91m'
    close = '\033[0m'

TRAINING_SIZE = 50000
DIGITS = 3
REVERSE = True

MAXLEN = DIGITS + 1 + DIGITS

chars = '0123456789+ '
ctable = CharacterTable(chars)

questions = []
expected = []
seen = set()
print('Generating data...')
while len(questions) < TRAINING_SIZE:
    f = lambda: int(''.join(np.random.choice(list('0123456789'))
                    for i in range(np.random.randint(1, DIGITS + 1))))
    a, b = f(), f()
    key = tuple(sorted((a, b)))
    if key in seen:
        continue
    seen.add(key)
    q = '{}+{}'.format(a, b)
    query = q + ' ' * (MAXLEN - len(q))
    ans = str(a + b)
    ans += ' ' * (DIGITS + 1 - len(ans))
    if REVERSE:
        query = query[::-1]
    questions.append(query)
    expected.append(ans)
print('Total addition questions:', len(questions))

print('Vectorization...')
x = np.zeros((len(questions), MAXLEN, len(chars)), dtype=np.bool)
y = np.zeros((len(questions), DIGITS + 1, len(chars)), dtype=np.bool)
for i, sentence in enumerate(questions):
    x[i] = ctable.encode(sentence, MAXLEN)
for i, sentence in enumerate(expected):
    y[i] = ctable.encode(sentence, DIGITS + 1)

indices = np.arange(len(y))
np.random.shuffle(indices)
x = x[indices]
y = y[indices]

split_at = len(x) - len(x) // 10
(x_train, x_val) = x[:split_at], x[split_at:]
(y_train, y_val) = y[:split_at], y[split_at:]

print('Training Data:')
print(x_train.shape)
print(y_train.shape)

print('Validation Data:')
print(x_val.shape)
print(y_val.shape)

HIDDEN_SIZE = 128
BATCH_SIZE = 128
LAYERS = 1

def addition_rnn_neural_network(inputs, labels):
    print('Build model...')

    # input_shape=(None, num_feature).
    _, hidden, _ = basic_lstm(inputs, None, None, HIDDEN_SIZE)
    expand_hidden = fluid.layers.expand(hidden[0], expand_times=[1, DIGITS + 1])
    outputs = fluid.layers.reshape(expand_hidden, shape=[BATCH_SIZE, DIGITS + 1, HIDDEN_SIZE])

    for _ in range(LAYERS):

        # outputs, _, _ = fluid.layers.lstm(outputs, init_h, init_c, MAXLEN, HIDDEN_SIZE, num_layers=1)
        outputs, _, _ = basic_lstm(outputs, None, None, HIDDEN_SIZE)

    probs = fluid.layers.fc(input=outputs, size=len(chars), act='softmax', num_flatten_dims=2)

    loss = fluid.layers.cross_entropy(input=probs, label=labels, soft_label=True)
    avg_loss = fluid.layers.mean(loss)
    preds = fluid.layers.reshape(probs, shape=[BATCH_SIZE * (DIGITS + 1), len(chars)])
    labs = fluid.layers.reshape(fluid.layers.argmax(labels, axis=-1), shape=[BATCH_SIZE * (DIGITS + 1), 1])
    accuracy = fluid.layers.accuracy(preds, labs)
    return avg_loss, accuracy

def train(args):
    if args.use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    def reader_creater(inputs, labels):
        def reader():
            for input, label in zip(inputs, labels):
                yield input, label

        return reader

    if args.enable_ce:
        train_reader = paddle.batch(reader_creater(x_train, y_train), batch_size=BATCH_SIZE, drop_last=True)
        test_reader = paddle.batch(reader_creater(x_val, y_val), batch_size=BATCH_SIZE, drop_last=True)
        startup_program.random_seed = 90
        main_program.random_seed = 90
    else:
        train_reader = paddle.batch(paddle.reader.shuffle(
            reader_creater(x_train, y_train), buf_size=5000), batch_size=BATCH_SIZE, drop_last=True)
        test_reader = paddle.batch(reader_creater(x_val, y_val), batch_size=BATCH_SIZE, drop_last=True)

    inputs = fluid.layers.data(name='inputs', shape=[BATCH_SIZE, 7, 12], dtype='float32', append_batch_size=False)
    labels = fluid.layers.data(name='labels', shape=[BATCH_SIZE, 4, 12], dtype='float32', append_batch_size=False)
    net_conf = addition_rnn_neural_network

    avg_loss, accuracy = net_conf(inputs, labels)


    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(avg_loss)

    def train_test(train_test_program, train_test_feed, train_test_reader):
        loss_set = []
        accuracy_set = []
        for test_data in train_test_reader():
            loss_np, accuracy_np = exe.run(
                program=train_test_program,
                feed=train_test_feed.feed(test_data),
                fetch_list=[avg_loss, accuracy])
            loss_set.append(float(loss_np))
            accuracy_set.append(float(accuracy_np))
        # get test acc and loss
        loss_mean = np.array(loss_set).mean()
        accuracy_mean = np.array(accuracy_set).mean()
        return loss_mean, accuracy_mean

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[inputs, labels], place=place)
    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(200)]

    lists = []
    step = 0
    for epoch_id in epochs:
        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(
                main_program,
                feed=feeder.feed(data),
                fetch_list=[avg_loss, accuracy])
            if step % 100 == 0:
                print("Pass %d, Epoch %d, avg_loss %f, accuracy %f" % (step, epoch_id,
                                                      metrics[0], metrics[1]))
            step += 1

        # test for epoch
        avg_loss_val, accuracy_val = train_test(
            train_test_program=test_program,
            train_test_reader=test_reader,
            train_test_feed=feeder)

        print("Test with Epoch %d, avg_loss_val: %s, accuracy_val: %s" %
              (epoch_id, avg_loss_val, accuracy_val))
        lists.append((epoch_id, avg_loss_val, accuracy_val))


    # find the best pass
    best = sorted(lists, key=lambda list: float(list[2]))[0]
    print('Best epoch is %s, accuracy_val is %s' % (best[0], best[2]))

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
