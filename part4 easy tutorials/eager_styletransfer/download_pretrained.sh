#!/bin/bash

if [ ! -d "./pretrained" ] 
then
	wget https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.tar
	tar -xf VGG19_pretrained.tar
	mkdir pretrained
	cd pretrained
	mkdir block1 block2 block3 block4 block5
	cd block1
	mkdir conv_block_2_0
	cd ..
	cd block2
	mkdir conv_block_2_0
	cd ..
	cd block3
	mkdir conv_block_4_0
	cd ..
	cd block4
	mkdir conv_block_4_0
	cd ..
	cd block5
	mkdir conv_block_4_0
	cd ../../VGG19_pretrained

	mv conv1_1_weights Conv2D_0.1_weights
	mv Conv2D_0.1_weights ../pretrained/block1/conv_block_2_0
	mv conv1_2_weights Conv2D_1.2_weights
	mv Conv2D_1.2_weights ../pretrained/block1/conv_block_2_0

	mv conv2_1_weights Conv2D_0.1_weights
	mv Conv2D_0.1_weights ../pretrained/block2/conv_block_2_0
	mv conv2_2_weights Conv2D_1.2_weights
	mv Conv2D_1.2_weights ../pretrained/block2/conv_block_2_0

	mv conv3_1_weights Conv2D_0.1_weights
	mv Conv2D_0.1_weights ../pretrained/block3/conv_block_4_0
	mv conv3_2_weights Conv2D_1.2_weights
	mv Conv2D_1.2_weights ../pretrained/block3/conv_block_4_0
	mv conv3_3_weights Conv2D_2.3_weights
	mv Conv2D_2.3_weights ../pretrained/block3/conv_block_4_0
	mv conv3_4_weights Conv2D_3.4_weights
	mv Conv2D_3.4_weights ../pretrained/block3/conv_block_4_0

	mv conv4_1_weights Conv2D_0.1_weights
	mv Conv2D_0.1_weights ../pretrained/block4/conv_block_4_0
	mv conv4_2_weights Conv2D_1.2_weights
	mv Conv2D_1.2_weights ../pretrained/block4/conv_block_4_0
	mv conv4_3_weights Conv2D_2.3_weights
	mv Conv2D_2.3_weights ../pretrained/block4/conv_block_4_0
	mv conv4_4_weights Conv2D_3.4_weights
	mv Conv2D_3.4_weights ../pretrained/block4/conv_block_4_0

	mv conv5_1_weights Conv2D_0.1_weights
	mv Conv2D_0.1_weights ../pretrained/block5/conv_block_4_0
	mv conv5_2_weights Conv2D_1.2_weights
	mv Conv2D_1.2_weights ../pretrained/block5/conv_block_4_0
	mv conv5_3_weights Conv2D_2.3_weights
	mv Conv2D_2.3_weights ../pretrained/block5/conv_block_4_0
	mv conv5_4_weights Conv2D_3.4_weights
	mv Conv2D_3.4_weights ../pretrained/block5/conv_block_4_0

	cd ..
	rm -rf VGG19_pretrained
	rm -rf VGG19_pretrained.tar
fi