#!/usr/bin/env python
# coding=utf-8
import numpy as np
import nltk
# nltk.download('punkt')
from nltk.stem.lancaster import LancasterStemmer
# There are a lot a differents algorithms:  Porter, Lancaster etc
# https://www.nltk.org/api/nltk.stem.html
stemmer = LancasterStemmer()

def tokenize(sentence):
    """
    split sentence into array of words
    input : string sentence
    output : array
    Ex:
        "I love Watermelon" --> ["I", "love", "Watermelon"]
    """
    return nltk.word_tokenize(sentence)


def stem(word):
    """
    stemming = find the root form of the word
    examples:
    words = ["organize", "organizes", "organizing"]
    words = [stem(w) for w in words]
    -> ["organ", "organ", "organ"]
    """
    return stemmer.stem(word.lower())


def bag_of_words(tokenized_sentence, words):
    """
    return bag of words array:
    1 for each known word that exists in the sentence, 0 otherwise
    example:
    sentence = ["hello", "how", "are", "you"]
    words = ["hi", "hello", "I", "you", "bye", "thank", "cool"]
    bog   = [  0 ,    1 ,    0 ,   1 ,    0 ,    0 ,      0]
    """
    # stem each word
    sentence_words = [stem(word) for word in tokenized_sentence]
    # initialize bag with 0 for each word
    bag = np.zeros(len(words), dtype=np.float32)
    for idx, w in enumerate(words):
        if w in sentence_words: 
            bag[idx] = 1

    return bag
