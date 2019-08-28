#!/bin/bash

PRETRAINED_DIR="./pretrained"

if [ ! -d "$PRETRAINED_DIR" ]; then
	echo "Download pretrained data..."
	wget https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar
	tar -xjvf InceptionV4_pretrained.tar
	mv InceptionV4_pretrained pretrained
	rm -f InceptionV4_pretrained.tar
fi