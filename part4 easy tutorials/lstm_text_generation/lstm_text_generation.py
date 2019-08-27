from __future__ import print_function
import numpy as np
import random
import sys
import io
import argparse

from data_utils import get_file
import paddle
import paddle.fluid as fluid
from paddle.fluid.contrib.layers import basic_lstm, BasicLSTMUnit

path, _ = get_file(
    'nietzsche.txt',
    origin='https://s3.amazonaws.com/text-datasets/nietzsche.txt')
with io.open(path, encoding='utf-8') as f:
    text = f.read().lower()
print('corpus length:', len(text))

chars = sorted(list(set(text)))
print('total chars:', len(chars))
char_indices = dict((c, i) for i, c in enumerate(chars))
indices_char = dict((i, c) for i, c in enumerate(chars))

BATCH_SIZE = 128
# cut the text in semi-redundant sequences of maxlen characters
maxlen = 40
step = 3
sentences = []
next_chars = []
for i in range(0, len(text) - maxlen, step):
    sentences.append(text[i: i + maxlen])
    next_chars.append(text[i + maxlen])
print('nb sequences:', len(sentences))

print('Vectorization...')
x = np.zeros((len(sentences), maxlen, len(chars)), dtype=np.bool)
y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        x[i, t, char_indices[char]] = 1
    y[i, char_indices[next_chars[i]]] = 1

def lstm_text_generation_neural_network(sentences, next_chars=None):
    print('Build model...')

    _, hidden, _ = basic_lstm(sentences, None, None, hidden_size=128)
    preds = fluid.layers.fc(input=hidden[0], size=len(chars), act='softmax')

    # loss
    loss = fluid.layers.cross_entropy(input=preds, label=next_chars, soft_label=True)
    avg_loss = fluid.layers.mean(loss)
    label = fluid.layers.reshape(fluid.layers.argmax(next_chars, axis=-1), shape=[-1, 1])
    accuracy = fluid.layers.accuracy(input=preds, label=label)

    return preds, avg_loss, accuracy

def sample(preds, temperature=1.0):
    # helper function to sample an index from a probability array
    preds = np.asarray(preds).astype('float64')[0]
    preds = np.log(preds) / temperature
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    probas = np.random.multinomial(1, preds, 1)
    return np.argmax(probas)


def train(args):
    if args.use_cuda and not fluid.core.is_compiled_with_cuda():
        return

    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    def reader_creater(sentences, next_chars):
        def reader():
            for sentence, next_char in zip(sentences, next_chars):
                yield sentence, next_char
        return reader

    if args.enable_ce:
        train_reader = paddle.batch(reader_creater(x, y), batch_size=BATCH_SIZE)
        startup_program.random_seed = 90
        main_program.random_seed = 90
    else:
        train_reader = paddle.batch(paddle.reader.shuffle(
            reader_creater(x, y), buf_size=1000), batch_size=BATCH_SIZE)

    sentences = fluid.layers.data(name="sentences", shape=[maxlen, len(chars)], dtype='float32')
    next_chars = fluid.layers.data(name="next_chars", shape=[len(chars)], dtype='float32')

    net_conf = lstm_text_generation_neural_network

    preds, avg_loss, accuracy = net_conf(sentences, next_chars)

    test_program = main_program.clone(for_test=True)
    optimizer = fluid.optimizer.RMSProp(learning_rate=0.01)
    optimizer.minimize(avg_loss)

    def train_test(train_test_program, epoch):
        # Function invoked at end of each epoch. Prints generated text.
        print()
        print('----- Generating text after Epoch: %d' % epoch)

        start_index = random.randint(0, len(text) - maxlen - 1)
        for diversity in [0.2, 0.5, 1.0, 1.2]:
            print('----- diversity:', diversity)

            generated = ''
            sentence = text[start_index: start_index + maxlen]
            generated += sentence
            print('----- Generating with seed: "' + sentence + '"')
            sys.stdout.write(generated)

            for i in range(400):
                x_pred = np.zeros((1, maxlen, len(chars)), dtype="float32")
                for t, char in enumerate(sentence):
                    x_pred[0, t, char_indices[char]] = 1.

                y_pred = np.zeros((1, len(chars)), dtype="float32")

                metrics = exe.run(
                    train_test_program,
                    feed={"sentences":x_pred, "next_chars":y_pred},
                    fetch_list=[preds])

                next_index = sample(metrics[0], diversity)
                next_char = indices_char[next_index]

                generated += next_char
                sentence = sentence[1:] + next_char

                sys.stdout.write(next_char)
                sys.stdout.flush()
            print()

    place = fluid.CUDAPlace(0) if args.use_cuda else fluid.CPUPlace()

    exe = fluid.Executor(place)

    feeder = fluid.DataFeeder(feed_list=[sentences, next_chars], place=place)
    exe.run(startup_program)
    epochs = [epoch_id for epoch_id in range(50)]


    step = 0
    for epoch_id in epochs:

        for step_id, data in enumerate(train_reader()):
            metrics = exe.run(
                main_program,
                feed=feeder.feed(data),
                fetch_list=[preds, avg_loss, accuracy])
            if step % 200 == 0:
                print("Pass %d, Epoch %d, loss %f" % (step, epoch_id, metrics[1]))
            step += 1

        print("Epoch %d, loss %f" % (epoch_id, metrics[1]))

        # test for epoch
        train_test(test_program, epoch_id)

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
