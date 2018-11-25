#!/usr/bin/env bash

# install required python packages
sudo apt-get update
apt-get install unzip
apt-get install python3
apt-get install python3-pip
pip3 install numpy
pip3 install matplotlib
pip3 install spacy
pip3 install torch
pip3 install json
python3 -m spacy download en

wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O ./train-v2.0.json
wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O ./dev-v2.0.json

wget http://nlp.stanford.edu/data/glove.6B.zip -O ./glove.6B.300d.zip
unzip ./glove.6B.300d.zip
wget https://github.com/nmrksic/counter-fitting/blob/master/word_vectors/counter-fitted-vectors.txt.zip?raw=true -O ./counter-fitted-vectors.txt.zip
unzip ./counter-fitted-vectors.txt.zip


