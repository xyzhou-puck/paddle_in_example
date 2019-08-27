import tarfile
import numpy as np
import re

import paddle
import paddle.fluid as fluid
import paddle.fluid.layers as layers
from paddle.fluid.contrib.layers import basic_lstm as basic_lstm


def tokenize(sent):
    
    return [x.strip() for x in re.split(r'(\W+)?', sent) if x.strip()]

def parse_stories(lines, only_supporting=False):

    data = []
    story = []
    for line in lines:
        line = line.decode('utf-8').strip()
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


def get_stories(f, only_supporting=False, max_length=None):
    
    data = parse_stories(f.readlines(), only_supporting=only_supporting)
    flatten = lambda data: reduce(lambda x, y: x + y, data)
    data = [(flatten(story), q, answer) for story, q, answer in data
        if not max_length or len(flatten(story)) < max_length]
    return data

def pad_sequences(input, maxlen):
    
    output = []

    for instance in input:
        if len(instance) >= maxlen:
            true_instance = instance[0:maxlen]
        else:
            true_instance = [0] * (maxlen - len(instance))
            true_instance.extend(instance)

        output.append(true_instance)

    return np.array(output).astype("int64")

def vectorize_stories(data):
    inputs, queries, answers = [], [], []
    for story, query, answer in data:
        inputs.append([word_idx[w] for w in story])
        queries.append([word_idx[w] for w in query])
        answers.append(word_idx[answer])
    
    np_inputs = pad_sequences(inputs, maxlen=story_maxlen)
    np_inputs = np.expand_dims(np_inputs, axis = -1).astype("int64")

    np_queries = pad_sequences(queries, maxlen=query_maxlen)
    np_queries = np.expand_dims(np_queries, axis = -1).astype("int64")
    
    np_answers = np.array(answers)
    np_answers = np.expand_dims(np_answers, axis = -1).astype("int64")

    return (np_inputs, np_queries, np_answers)

challenges = {
    'single_supporting_fact_10k': 'tasks_1-20_v1-2/en-10k/qa1_'
        'single-supporting-fact_{}.txt',
    'two_supporting_facts_10k': 'tasks_1-20_v1-2/en-10k/qa2_'
        'two-supporting-facts_{}.txt',
}

challenge_type = 'single_supporting_fact_10k'
challenge = challenges[challenge_type]

print('Extracting stories for the challenge:', challenge_type)
path = "./babi_tasks_1-20_v1-2.tar.gz"
with tarfile.open(path) as tar:
    train_stories = get_stories(tar.extractfile(challenge.format('train')))
    test_stories = get_stories(tar.extractfile(challenge.format('test')))

vocab = set()
for story, q, answer in train_stories + test_stories:
    vocab |= set(story + q + [answer])
vocab = sorted(vocab)

vocab_size = len(vocab) + 1
story_maxlen = max(map(len, (x for x, _, _ in train_stories + test_stories)))
query_maxlen = max(map(len, (x for _, x, _ in train_stories + test_stories)))

print('-')
print('Vocab size:', vocab_size, 'unique words')
print('Story max length:', story_maxlen, 'words')
print('Query max length:', query_maxlen, 'words')
print('Number of training stories:', len(train_stories))
print('Number of test stories:', len(test_stories))
print('-')
print('Here\'s what a "story" tuple looks like (input, query, answer):')
print(train_stories[0])
print('-')
print('Vectorizing the word sequences...')

word_idx = dict((c, i + 1) for i, c in enumerate(vocab))
inputs_train, queries_train, answers_train = vectorize_stories(train_stories)
inputs_test, queries_test, answers_test = vectorize_stories(test_stories)

print('-')
print('inputs: integer tensor of shape (samples, max_length)')
print('inputs_train shape:', inputs_train.shape)
print('inputs_test shape:', inputs_test.shape)
print('-')
print('queries: integer tensor of shape (samples, max_length)')
print('queries_train shape:', queries_train.shape)
print('queries_test shape:', queries_test.shape)
print('-')
print('answers: binary (1 or 0) tensor of shape (samples, vocab_size)')
print('answers_train shape:', answers_train.shape)
print('answers_test shape:', answers_test.shape)
print('-')
print('Compiling...')

input_sequence = layers.data(name = "story", dtype = "int64", shape = [-1, story_maxlen, 1])
question = layers.data(name = "query", dtype = "int64", shape = [-1, query_maxlen, 1])
true_answer = layers.data(name = "true_answer", dtype = "int64", shape = [-1, 1])

input_encoder_m = layers.embedding(input = input_sequence, size = [vocab_size, 64])
input_encoder_m = layers.dropout(input_encoder_m, 0.3)

input_encoder_c = layers.embedding(input = input_sequence, size = [vocab_size, query_maxlen])
input_encoder_c = layers.dropout(input_encoder_c, 0.3)

question_encoder = layers.embedding(input = input_sequence, size = [vocab_size, 64])
question_encoder = layers.dropout(question_encoder, 0.3)

match = layers.elementwise_mul(input_encoder_m, question_encoder)
response = layers.softmax(match, axis = -1)

answer = layers.concat([response, question_encoder], axis = -1)

_, _, answer = basic_lstm(answer, None, None, 32)
answer = layers.transpose(answer, perm = (1, 0, 2))
answer = layers.reshape(answer, shape = [-1, 32])

answer = layers.dropout(answer, 0.3)
answer = layers.fc(answer, size = vocab_size, act = "softmax")

loss = layers.cross_entropy(answer, true_answer)
loss = layers.reduce_mean(loss)

optimizer = fluid.optimizer.AdamOptimizer(learning_rate = 0.01)
optimizer.minimize(loss)

exe = fluid.Executor(fluid.CPUPlace())

exe.run(fluid.default_startup_program())

print('Training...')

batch_size = 32

i = 0
step = 0
while i < answers_train.shape[0]:
    curr_inputs_train = inputs_train[i:i+batch_size, :, :]
    curr_queries_train = queries_train[i:i+batch_size, :, :]
    curr_answers_train = answers_train[i:i+batch_size, :]

    curr_loss = exe.run(fluid.default_main_program(), 
        feed = {
            "story": curr_inputs_train, 
            "query": curr_queries_train, 
            "true_answer": curr_answers_train},
        
        fetch_list = [loss.name])

    print("step %d, loss %.3f" % (step, curr_loss[0][0]))
    i += batch_size
    step += 1





