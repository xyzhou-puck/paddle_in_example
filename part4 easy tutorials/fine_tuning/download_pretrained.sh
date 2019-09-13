#!/bin/bash

if [ ! -d "./pretrained" ] 
then
	wget https://paddle-imagenet-models-name.bj.bcebos.com/MobileNetV3_small_x1_0_pretrained.tar
	tar -xf MobileNetV3_small_x1_0_pretrained.tar
	mv MobileNetV3_small_x1_0_pretrained pretrained
fi