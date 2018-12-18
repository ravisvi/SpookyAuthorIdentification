import sys
from lang_model_lstm import LSTMLanguageModel
import numpy as np
import string
import cPickle as pickle
import re
from DataGenerator import DataGenerator

class DataGeneratorVector(DataGenerator):
    def __init__(self, vector_size=1):
        super(DataGeneratorVector, self).__init__(vector_size=vector_size)
        self.embedding_size = 1

    def load_embeddings(self, fname):
        vec_file = open(fname, 'r')
        self.glove = {}
        for line in vec_file:
            split = line.strip().split()
            split[0] = split[0].lower().decode('utf-8') # all tokens are lowercased in our dataset
            vector = np.array(split[1:], dtype="float32")
            self.embedding_size = vector.shape[0]
            try:
                self.glove[split[0]] = vector
            except Exception, e:
                print e
                continue
        self.glove["<s>"] = np.random.uniform(low=-0.005,high=0.005,size=(self.embedding_size,))
        self.glove["</s>"] = np.random.uniform(low=-0.005,high=0.005,size=(self.embedding_size,))
        print "vector size: " + str(self.vector_size)

    def parse_text(self, word_list):
        curr_seq = []
        curr_target = []
        flag = True
        for word in word_list:
            word = word.replace("'","")
            if word not in self.glove:
                if word == "": continue
                print "word: " + word + " not found in word embeddings, skipping sentence"
                flag = False
                break
            if word not in self.word_idx:
                self.word_idx[word] = self.count
                self.reverse_idx[self.count] = word
                self.count += 1
            curr_seq.append(self.word_idx[word])
            curr_target.append(self.word_idx[word])
        if flag:
            self.input_seq.append(self.word_idx["<s>"])
            self.target_seq.append(self.word_idx["<s>"])
            self.input_seq.extend(curr_seq)
            self.target_seq.extend(curr_target)
            self.input_seq.append(self.word_idx["</s>"])
            self.target_seq.append(self.word_idx["</s>"])

    def get_embedding_weights(self):
        embedding_matrix = np.zeros((len(self.word_idx), self.embedding_size))
        for word in self.word_idx:
            idx = self.word_idx[word]
            vector = self.glove[word]
            embedding_matrix[idx] = vector
        return embedding_matrix
