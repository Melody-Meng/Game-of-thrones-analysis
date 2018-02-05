#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import re
import pandas as pd
import numpy as np
import nltk
from nltk.stem.snowball import SnowballStemmer
from sklearn.cluster import KMeans

class GetData:
    # extract keywords from episodes
    def word_extract(f):
        # tokenize 
        tokens = nltk.word_tokenize(f)
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(["'s", "n't", "'m", "'d", "us","would","know","one","go","want","come","like","get","veri","well","thing","king","re","ve","ever","still"])
        stemmer = SnowballStemmer("english")
        # extract stemmers
        word = []
        for token in tokens:
            if re.search('[a-zA-Z]', token):
                word.append(token)
        word = [s.lower() for s in word if s.lower() not in stopwords]
        stems = [stemmer.stem(t) for t in word]
        stems = [s.lower() for s in stems if s.lower() not in stopwords]
        return stems
       
    # extract keywords from paragraphs
    def para_data(f):
        # separate paragraphs
        para = f.split("\n\n")
        para[:] = (value for value in para if value != '\t')

        # Separate and tokenize
        Records = []
        Words = []
        Time = []

        for line in range (len(para)):
            lines = word_tokenize(para[line])

            all_word = []
            for token in lines:
                if re.search('[a-zA-Z]', token):
                    all_word.append (token)
            Words.append(all_word)

        # Stem and remove stopwords
        stemmer = SnowballStemmer("english")
        stopwords = nltk.corpus.stopwords.words('english')
        stopwords.extend(["'s", "n't", "'m", "'d", "us","would","know","one","go","want","come","like","king","north","army","father","need","think","armi","dead","back","lord"])
        for i in range (len(Words)):
            Words[i] = [stemmer.stem(t) for t in Words[i]]
            Words[i] = [s.lower() for s in Words[i] if s.lower() not in stopwords]
        
        
        return Words

    def most_100(f):
        return nltk.FreqDist(f).most_common(100)
    
    def unique(f):
        return list(nltk.FreqDist(f).keys())
