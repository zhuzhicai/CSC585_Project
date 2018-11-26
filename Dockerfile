# base image
FROM python:3-onbuild

# copy data to working directory
WORKDIR /app
COPY . /app

# install dependencies
RUN apt-get update
RUN apt-get install unzip
RUN pip install -r requirements.txt
RUN python3 -m spacy download en
RUN wget https://rajpurkar.github.io/SQuAD-explorer/dataset/train-v2.0.json -O ./train-v2.0.json
RUN wget https://rajpurkar.github.io/SQuAD-explorer/dataset/dev-v2.0.json -O ./dev-v2.0.json
RUN wget https://github.com/nmrksic/counter-fitting/blob/master/word_vectors/counter-fitted-vectors.txt.zip?raw=true -O ./counter-fitted-vectors.txt.zip
RUN unzip ./counter-fitted-vectors.txt.zip

# training
RUN python3 train.py train-v2.0.json counter-fitted-vectors.txt

# THE RESULT models and images are in current directory.


