#!/bin/bash
echo 'mnist transfer cnn'
python3 train.py
sleep .5
python3 fine_tune.py