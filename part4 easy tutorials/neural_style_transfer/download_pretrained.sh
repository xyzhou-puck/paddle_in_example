#!/bin/bash

if [ ! -d "./pretrained" ] 
then
	# wget https://paddle-imagenet-models-name.bj.bcebos.com/VGG19_pretrained.tar
	tar -xf VGG19_pretrained.tar
	mv VGG19_pretrained pretrained
	rm -rf ./pretrained/fc*
	rm -rf VGG19_pretrained.tar
fi