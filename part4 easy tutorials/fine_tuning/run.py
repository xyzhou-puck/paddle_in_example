import os, shutil
import wget, zipfile
import cv2
import warnings
import random
import numpy as np
import paddle
import paddle.fluid as fluid
from PIL import Image
from progress.bar import Bar
from math import floor
import MobileNetV3 
import reader

warnings.filterwarnings('ignore')

# Download data -----------------------------------------------------------

print('downloading...')
url = 'https://download.microsoft.com/download/3/E/1/3E1C3F21-ECDB-4869-8368-6DEBA77B919F/kagglecatsanddogs_3367a.zip'
wget.download(url, './cats-dogs.zip')

# Pre-processing ----------------------------------------------------------

print('unziping...')
with zipfile.ZipFile('./cats-dogs.zip', 'r') as zip_ref:
    zip_ref.extractall('./data-raw')

# We will organize images in the following structure:
# data/
#     train/
#          Cat/
#          Dog/
#     validation
#          Cat/
#          Dog/
#     test/
#          images/
#

# some images are corrupt and we exclude them
# this will make sure all images can be read.
print('checking...')
for path, subdirs, files in os.walk('./data-raw/PetImages'):
    for filename in files:
        if filename.endswith('.jpg'):
            filepath = os.path.join(path, filename)
            try:
                img = Image.open(filepath)
                img.verify()
            except (IOError, SyntaxError) as e:
                os.remove(filepath)
                print('removed image: ', filename)
                continue

            img = cv2.imread(filepath)
            if img is None:
                os.remove(filepath)
                print('removed image: ', filename)

# re-list all imgs
all_imgs = []
for path, subdirs, files in os.walk('./data-raw/PetImages'):
    for filename in files:
        if filename.endswith('.jpg'):
            all_imgs.append(os.path.join(path, filename))

random.shuffle(all_imgs)
size_2, size_4 = floor(len(all_imgs) / 2), floor(len(all_imgs) / 4)
training_imgs = all_imgs[:size_2]
validation_imgs = all_imgs[size_2 : size_2 + size_4]
testing_imgs = all_imgs[size_2 + size_4:]

# create directory structure
os.makedirs('./data/train/Cat')
os.makedirs('./data/train/Dog')
os.makedirs('./data/validation/Cat')
os.makedirs('./data/validation/Dog')
os.makedirs('./data/test/images')

# copy training images
print('copying...')
for training_img in training_imgs:
    pet = (training_img.find('Cat') < 0) and 'Dog' or 'Cat'
    shutil.copy(training_img, './data/train/' + pet)

# copy valid images
for validation_img in validation_imgs:
    pet = (validation_img.find('Cat') < 0) and 'Dog' or 'Cat'
    shutil.copy(validation_img, './data/validation/' + pet)

# copy testing imgs
for testing_img in testing_imgs:
    shutil.copy(testing_img, './data/test/images')

# Image flow --------------------------------------------------------------

print('preparing data...')

train_data = reader.train('./data/train/', 224, 20, 0.2, 0.2, True)
validation_data = reader.val('./data/validation/', 224)

training_image_flow = paddle.batch(
    reader.reader_creator(train_data, 'train'), batch_size=100, drop_last=True)
validation_image_flow = paddle.batch(
    reader.reader_creator(validation_data, 'val'), batch_size=100, drop_last=True)

# Model -------------------------------------------------------------------

print('loading model...')

place = fluid.CUDAPlace(1)
exe = fluid.Executor(place)

input_tensor = fluid.layers.data(
    name='input', shape=(-1, 3, 224, 224), dtype='float32')
label_tensor = fluid.layers.data(
    name='label', shape=(-1, 1), dtype='int64')

mob = MobileNetV3.MobileNetV3(include_top=False)
mob_out = mob.net(input=input_tensor)

fluid.io.load_params(exe, './pretrained')

fc1 = fluid.layers.fc(mob_out, 256, act='relu', name='top_fc1')
dropout1 = fluid.layers.dropout(fc1, 0.2, name='top_dropout1')
fc2 = fluid.layers.fc(dropout1, 2, act='sigmoid', name='top_fc2')

loss = fluid.layers.cross_entropy(input=fc2, label=label_tensor)
avg_loss = fluid.layers.mean(loss)
acc = fluid.layers.accuracy(fc2, label_tensor)

startup_program = fluid.default_startup_program()
main_program = fluid.default_main_program()
test_program = main_program.clone(for_test=True)

optimizer = fluid.optimizer.Adam(learning_rate=0.001)
optimizer.minimize(avg_loss)

exe.run(startup_program)

feeder = fluid.DataFeeder(feed_list=[input_tensor, label_tensor], place=place)

for var in main_program.list_vars():
    if fluid.io.is_parameter(var) and not (var.name.startswith('top')):
        var.trainable = False

print('\ntraining...')
for epoch in range(1):
    
    num_batches = int(np.ceil(len(train_data[0]) / float(100))) - 1
    progress_bar = Bar('', max=num_batches)
    
    loss_set, acc_set = [], []
    for step, data in enumerate(training_image_flow()):

        fetch_out = exe.run(
            program=main_program,
            feed=feeder.feed(data),
            fetch_list=[avg_loss.name, acc.name])

        loss_set.append(fetch_out[0][0])
        acc_set.append(fetch_out[1][0])
        progress_bar.next()
    loss_set = np.array(loss_set).mean()
    acc_set = np.array(acc_set).mean()
    progress_bar.finish()
    print('train loss: %f, acc: %f' % (loss_set, acc_set))

print('validating...')
loss_set, acc_set = [], []
for step, data in enumerate(validation_image_flow()):
    fetch_out = exe.run(
        program=test_program,
        feed=feeder.feed(data),
        fetch_list=[avg_loss.name, acc.name])

    loss_set.append(fetch_out[0][0])
    acc_set.append(fetch_out[1][0])
loss_set = np.array(loss_set).mean()
acc_set = np.array(acc_set).mean()
print('val loss: %f, acc: %f' % (loss_set, acc_set))

# now top layers weights are fine, we can unfreeze the lower layer weights.

for var in main_program.list_vars():
    if fluid.io.is_parameter(var) and not (var.name.startswith('top')):
        var.trainable = True

print('\ntraining...')
for epoch in range(1):

    num_batches = int(np.ceil(len(train_data[0]) / float(100))) - 1
    progress_bar = Bar('', max=num_batches)

    loss_set, acc_set = [], []
    for step, data in enumerate(training_image_flow()):

        fetch_out = exe.run(
            program=main_program,
            feed=feeder.feed(data),
            fetch_list=[avg_loss.name, acc.name])

        loss_set.append(fetch_out[0][0])
        acc_set.append(fetch_out[1][0])
        progress_bar.next()
    loss_set = np.array(loss_set).mean()
    acc_set = np.array(acc_set).mean()
    progress_bar.finish()
    print('train loss: %f, acc: %f' % (loss_set, acc_set))

print('validating...')
loss_set, acc_set = [], []
for step, data in enumerate(validation_image_flow()):
    fetch_out = exe.run(
        program=test_program,
        feed=feeder.feed(data),
        fetch_list=[avg_loss.name, acc.name])

    loss_set.append(fetch_out[0][0])
    acc_set.append(fetch_out[1][0])
loss_set = np.array(loss_set).mean()
acc_set = np.array(acc_set).mean()
print('val loss: %f, acc: %f' % (loss_set, acc_set))

# Generate predictions for test data --------------------------------------

test_data = reader.test('./data/test/', 224)
test_flow = paddle.batch(
    reader.reader_creator(test_data, 'test'), batch_size=100, drop_last=True)
testing_imgs, _ = test_data

def display(img):
    img = img[0] * 255.0
    img = np.clip(img, 0, 255).astype('uint8')
    img = np.transpose(img, (1, 2, 0))
    img = Image.fromarray(img, 'RGB')
    b, g, r = img.split()
    img = Image.merge("RGB", (r, g, b))
    img.show()

test_img = np.expand_dims(testing_imgs[1], axis=0)
print(test_img.shape)
pred = exe.run(
    program=test_program,
    feed={'input' : test_img, 'label' : np.array([[0]]).astype('int64')},
    fetch_list=[fc2.name])[0]
print('pred: ', ((np.argmax(pred) == 0) and 'Dog' or 'Cat'))
display(test_img)

# test_img = np.expand_dims(testing_imgs[5123], axis=0)
# pred = exe.run(
#     program=test_program,
#     feed={'input' : test_img, 'label' : np.array([[5123]]).astype('int64')},
#     fetch_list=[fc2])[0]
# print('pred: ', ((np.argmax(pred) == 0) and 'Dog' or 'Cat'))
# display(test_img)