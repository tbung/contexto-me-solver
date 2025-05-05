#!/usr/bin/env bash

wget -P data 'https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/google-10000-english-usa-no-swears.txt'
wget -P data 'https://raw.githubusercontent.com/first20hours/google-10000-english/refs/heads/master/20k.txt'
wget -P data 'https://nlp.stanford.edu/data/glove.840B.300d.zip'
unzip -d data data/glove.840B.300d.zip
