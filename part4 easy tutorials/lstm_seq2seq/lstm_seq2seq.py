from __future__ import print_function

import os
import argparse
from PIL import Image
import numpy as np
import paddle
import paddle.fluid as fluid
import paddle.fluid.framework as framework
from base_model import BaseModel
import reader

batch_size = 64 # Batch size for training.
num_epochs = 100  # Number of epochs to train for.
latent_dim = 256  # Latent dimensionality of the encoding space.
num_samples = 10000  # Number of samples to train on.
data_path = 'fra.txt'

def prepare_input(batch, epoch_id=0, with_lr=True):
    src_ids, src_mask, tar_ids, tar_mask = batch
    res = {}
    
    src_ids = src_ids.reshape((src_ids.shape[0], src_ids.shape[1], 1))
    in_tar = tar_ids[:, :-1]
    label_tar = tar_ids[:, 1:]

    in_tar = in_tar.reshape((in_tar.shape[0], in_tar.shape[1], 1))
    label_tar = label_tar.reshape((label_tar.shape[0], label_tar.shape[1], 1))
 
    res['src'] = src_ids
    res['tar'] = in_tar
    res['label'] = label_tar
    res['src_sequence_length'] = src_mask
    res['tar_sequence_length'] = tar_mask
 
    return res, np.sum(tar_mask)

def train():
    startup_program = fluid.default_startup_program()
    main_program = fluid.default_main_program()

    raw_data = reader.raw_data('fra.txt', num_samples=num_samples)    
    train_data = raw_data[0]
    data_vars = raw_data[1]

    model = BaseModel(hidden_size=latent_dim,
                      src_vocab_size=data_vars['num_encoder_tokens'],
                      tar_vocab_size=data_vars['num_decoder_tokens'],
                      batch_size=batch_size,
                      batch_first=True)

    loss = model.build_graph()

    optimizer = fluid.optimizer.Adam(learning_rate=0.001)
    optimizer.minimize(loss)

    place = fluid.CUDAPlace(0)
    exe = fluid.Executor(place)
    exe.run(framework.default_startup_program())

    ce_ppl = []
    for epoch_id in range(num_epochs):
        print("epoch ", epoch_id)
        
        train_data_iter = reader.get_data_iter(train_data, batch_size)
	
        total_loss = 0
        word_count = 0.0
        for batch_id, batch in enumerate(train_data_iter):

           

            input_data_feed, word_num = prepare_input(batch, epoch_id=epoch_id)
            fetch_outs = exe.run(feed=input_data_feed,
                                 fetch_list=[loss.name],
                                 use_program_cache=True)

            cost_train = np.array(fetch_outs[0])

            total_loss += cost_train * batch_size
            word_count += word_num    

            if batch_id > 0 and batch_id % batch_size == 0:
                print("  ppl", batch_id, np.exp(total_loss / word_count))
                ce_ppl.append(np.exp(total_loss / word_count))
                total_loss = 0.0
                word_count = 0.0

def main():
    train()

if __name__ == '__main__':
    main()










