import sys
from lang_model_lstm import LSTMLanguageModel
import numpy as np
import string

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
        if not author.startswith("EAP"): continue
        text = text.translate(None, string.punctuation).lower().split()
        data_object.parse_text(text)
    predictors, label = data_object.get_inputs()
    print(predictors.shape)
    print(label.shape)
    print(data_object.get_vocabulary_size())
    lm_object.train_model(data_object)


if __name__=="__main__":
    main("/Users/ambermadvariya/Documents/585/project/data/train.csv")
