#!/bin/bash

if [ ! -f "./fra.txt" ] 
then
	wget http://www.manythings.org/anki/fra-eng.zip
	unzip fra-eng.zip
	rm -rf _about.txt
	rm -rf fra-eng.zip
fi
