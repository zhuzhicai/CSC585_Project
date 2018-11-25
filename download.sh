#!/usr/bin/env bash

PWD=$(pwd)
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O ./train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O ./dev-v2.0.json

wget http://nlp.stanford.edu/data/glove.6B.zip -O ./glove.6B.300d.zip
unzip ./glove.6B.300d.zip -d ./


# install required python packages
# sudo apt-get update
# apt-get install python3
# apt-get install python-pip
# pip install numpy
# pip install matplotlib
# pip install spacy
# pip install pytorch
# pip install json
