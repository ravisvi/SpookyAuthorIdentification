import sys
from lang_model_lstm import LSTMLanguageModel
import numpy as np
import string
import cPickle as pickle
import re
from keras.preprocessing.text import Tokenizer
from DataGenerator import DataGenerator

class DataGeneratorSeq(DataGenerator):
    def __init__(self, vector_size=1):
        super(DataGeneratorSeq, self).__init__(vector_size=vector_size)

    def get_inputs(self):
        res = []
        for i in range(self.vector_size, len(self.input_seq)):
            seq = self.input_seq[i - self.vector_size: i]
            res.append(seq)

        num_samples = len(res)
        self.input_seq = np.asarray(res).reshape((num_samples, self.vector_size))
        return self.input_seq[:,:-1], self.input_seq[:,-1]