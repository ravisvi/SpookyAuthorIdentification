import sys
from lang_model_lstm import LSTMLanguageModel
import numpy as np
import string
import cPickle as pickle
import re
from keras.preprocessing.text import Tokenizer

class DataGenerator(object):
    def __init__(self, vector_size=1):
        self.word_idx = {}
        self.word_idx["."] = 0
        self.reverse_idx = {}
        self.reverse_idx[0] = "."
        self.count = 1
        self.input_seq = []
        self.target_seq = []
        self.vector_size = vector_size

    def parse_text(self, word_list):
        for word in word_list:
            if word not in self.word_idx:
                self.word_idx[word] = self.count
                self.reverse_idx[self.count] = word
                self.count += 1
            self.input_seq.append(self.word_idx[word])
            self.target_seq.append(self.word_idx[word])
        self.input_seq.append(self.word_idx["."])
        self.target_seq.append(self.word_idx["."])

    def get_inputs(self):
        if type(self.input_seq) is np.ndarray:
            return self.input_seq, self.target_seq
        else:
            print "Checking length: ", len(self.input_seq), len(self.target_seq)
            num_samples = len(self.input_seq) - 1
            self.input_seq = self.input_seq[:-1]
            self.input_seq = np.asarray(self.input_seq).reshape((num_samples, self.vector_size))
            self.target_seq = self.target_seq[1:]
            self.target_seq = np.asarray(self.target_seq).reshape((num_samples, 1))
            return self.input_seq, self.target_seq

    def get_vocabulary_size(self): 
        return len(self.word_idx)

    def get_sequence(self, words):
        result = []
        for word in words:
            if word not in self.word_idx:
                raise Exception("Word: " + word + " not found in word index. Model currently only generates text on vocabulary seen before")
            result.append(self.word_idx[word])
        return np.asarray(result).reshape((len(words), self.vector_size))

    def get_sentence(self, input_seq):
        result = ""
        for token in input_seq:
            if token not in self.reverse_idx:
                raise Exception("Token: " + str(token) + " not in reverse index")
            result = result + " " + self.reverse_idx[token]
        return result

    def get_word(self, idx):
        if idx not in self.reverse_idx:
            raise Exception("idx: " + str(idx) + "not in reverse index")
        return self.reverse_idx[idx]

    def get_vector_size(self):
        return self.vector_size