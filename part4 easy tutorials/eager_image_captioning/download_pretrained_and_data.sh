#!/bin/bash

# download Inception v4 pretrained weights
if [ ! -d "./pretrained" ] 
then
	echo "download Inception v4..."
	wget https://paddle-imagenet-models-name.bj.bcebos.com/InceptionV4_pretrained.tar
	tar -xf InceptionV4_pretrained.tar
	mv InceptionV4_pretrained pretrained
fi

# download coco 2014 annotations
if [ ! -d "./annotations"]
then
	echo "download coco 2014 annotations..."
	wget http://images.cocodataset.org/annotations/annotations_trainval2014.zip
	unzip ./annotations_trainval2014.zip
	mv annotations_trainval2014 annotations
fi

# download coco 2014 images
if [ ! -d "./train2014"]
then
	echo "download coco 2014 images..."
	echo "Caution: large download ahead! (13GB)"
	wget http://images.cocodataset.org/zips/train2014.zip
	unzip ./train2014.zip
fi
