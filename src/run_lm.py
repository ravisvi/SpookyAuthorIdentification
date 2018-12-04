import sys
from lang_model_lstm import LSTMLanguageModel
import numpy as np
import string
import cPickle as pickle

class DataGenerator(object):
    def __init__(self, input_size=1):
        self.input_size = input_size
        self.word_idx = {}
        self.reverse_idx = {}
        self.word_idx["<s>"] = 0
        self.reverse_idx[0] = "<s>"
        self.word_idx["</s>"] = 1
        self.reverse_idx[1] = "</s>"
        self.count = 2
        self.input_seq = []

    def parse_text(self, word_list):
        self.input_seq.append(0)
        for word in word_list:
            if word not in self.word_idx:
                self.word_idx[word] = self.count
                self.reverse_idx[self.count] = word
                self.count += 1
            self.input_seq.append(self.word_idx[word])
        self.input_seq.append(1)

    def get_inputs(self):
        input_matrix = self.input_seq[:-1]
        input_matrix = np.asarray(input_matrix).reshape((len(self.input_seq) - 1,1))
        targets = self.input_seq[1:]
        targets = np.asarray(targets).reshape((len(self.input_seq) - 1,1))
        return input_matrix, targets

    def get_vocabulary_size(self): 
        return len(self.word_idx)

    def get_sequence(self, words):
        result = []
        for word in words:
            if word not in self.word_idx:
                raise Exception("Word: " + word + " not found in word index. Model currently only generates text on vocabulary seen before")
            result.append(self.word_idx[word])
        return np.asarray(result).reshape((len(words),1))

    def get_sentence(self, input_seq):
        result = ""
        for token in input_seq:
            if token not in self.reverse_idx:
                raise Exception("Token: " + str(token) + " not in reverse index")
            result = result + " " + self.reverse_idx[token]
        return result

def main(fname):
    data_object = DataGenerator()
    lm_object = LSTMLanguageModel()
    infile = open(fname)
    first = True
    input_seq = []
    for line in infile:
        if first:
            first = False
            continue
        text_id, text, author = line.split('","')
        #if not author.startswith("EAP"): continue
        text = text.translate(None, string.punctuation).lower().split()
        data_object.parse_text(text)
    predictors, label = data_object.get_inputs()
    print(predictors.shape)
    print(label.shape)
    print(data_object.get_vocabulary_size())
    lm_object.train_model(data_object)
    return lm_object, data_object

def generate_sentence(lm_object, data_object, sentence, seq_len):
    sentence = sentence.translate(None, string.punctuation).lower().split()
    lstm_seq = data_object.get_sequence(sentence)
    res = lm_object.generate_text(lstm_seq, seq_len)
    print(data_object.get_sentence(res))


if __name__=="__main__":
    lm_object, data_object = main("../data/train.csv")
