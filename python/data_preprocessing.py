import itertools
import numpy as np
import sys
import nltk
import csv
from datetime import datetime
from utils import *
import matplotlib.pyplot as plt

vocabulary_size = 8000

unknow_word = "UNKNOW_WORD"
start_word = "START_WORD"
end_word = "END_WORD"

def data_preprocessing():
    print ("Start Proprocessing Input Data")
    with open("data/reddit-comments-2015-08.csv") as file:
        reader = csv.reader(file, skipinitialspace=True)
        sentences = itertools.chain(*[nltk.sent_tokenize(x[0].decode('utf-8').lower()) for x in reader])
        sentences = ["%s %s %s" % (start_word, x, end_word) for x in sentences]

    word_sentences = [nltk.word_tokenize(sentence) for sentence in sentences]
    word_freq = nltk.FreqDist(itertools.chain(*word_sentences))

    vocab = word_freq.most_common(vocabulary_size - 1)
    index_to_word = [x[0] for x in vocab]
    index_to_word.append(unknow_word)
    word_to_index = dict([(w,i) for i,w in enumerate(index_to_word)])

    # Replace all words not in our vocabulary with the unknown token
    for i, sent in enumerate(word_sentences):
        word_sentences[i] = [w if w in word_to_index else unknow_word for w in sent]

    # Create the training data
    x_train = np.asarray([[word_to_index[w] for w in sent[:-1]] for sent in word_sentences])
    y_train = np.asarray([[word_to_index[w] for w in sent[1:]] for sent in word_sentences])
    print ("Finishing Data Processing")
    return x_train, y_train



