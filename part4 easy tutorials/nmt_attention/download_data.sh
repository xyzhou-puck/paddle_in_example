#!/bin/bash

if [ ! -d "./data" ] 
then
	wget http://www.manythings.org/anki/nld-eng.zip
	mkdir data
	unzip ./nld-eng.zip -d ./data
	rm -f ./nld-eng.zip
fi