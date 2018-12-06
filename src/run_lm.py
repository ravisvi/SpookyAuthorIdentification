import sys
from lang_model_lstm import LSTMLanguageModel
import numpy as np
import string
import cPickle as pickle
import re

class DataGenerator(object):
    def __init__(self, input_size=1):
        self.input_size = input_size
        self.word_idx = {}
        self.reverse_idx = {}
        self.count = 0
        self.input_seq = []
        self.target_seq = []
        self.vector_size = 1

    def load_embeddings(self, fname):
        vec_file = open(fname, 'r')
        self.glove = {}
        for line in vec_file:
            split = line.strip().split()
            split[0] = split[0].lower().decode('utf-8') # all tokens are lowercased in our dataset
            vector = np.array(split[1:], dtype="float32")
            self.vector_size = vector.shape[0]
            try:
                self.glove[split[0]] = vector
            except Exception, e:
                print e
                continue
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
            curr_seq.append(self.glove[word])
            curr_target.append(self.word_idx[word])
        if flag:
            self.input_seq.extend(curr_seq)
            self.target_seq.extend(curr_target)

    def get_inputs(self):
        if type(self.input_seq) is np.ndarray:
            return self.input_seq, self.target_seq
        else:
            print "Checking length: ", len(self.input_seq), len(self.target_seq)
            num_samples = len(self.input_seq) - 1
            self.input_seq = self.input_seq[:-1]
            self.input_seq = np.asarray(self.input_seq).reshape((num_samples, 1, self.vector_size))
            self.target_seq = self.target_seq[1:]
            self.target_seq = np.asarray(self.target_seq).reshape((num_samples, 1))
            return self.input_seq, self.target_seq

    def get_vocabulary_size(self): 
        return len(self.word_idx)

    def get_sequence(self, words):
        result = []
        for word in words:
            if word not in self.glove:
                raise Exception("Word: " + word + " not found in word index. Model currently only generates text on vocabulary seen before")
            result.append(self.glove[word])
        return np.asarray(result).reshape((len(words), 1, self.vector_size))

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

    def get_vector(self, word):
        if word not in self.glove:
            raise Exception("word: " + str(word) + "not in glove")
        return self.glove[word]

    def get_vector_size(self):
        return self.vector_size

def main(glove_fname, fname):
    data_object = DataGenerator()
    lm_object = LSTMLanguageModel(layer_size=1,epochs=1)
    infile = open(fname)
    first = True
    input_seq = []
    data_object.load_embeddings(glove_fname)
    for line in infile:
        if first:
            first = False
            continue
        text_id, text, author = line.split('","')
        if not author.startswith("EAP"): continue
        text = text.replace("-"," ")
        text = text.translate(None, string.punctuation).lower().split()
        text.append(".")
        #text = re.findall(r"[\w']+|[.,!?;]", text.lower())
        data_object.parse_text(text)
    predictors, label = data_object.get_inputs()
    lm_object.train_model(data_object)
    return lm_object, data_object

def generate_sentence(lm_object, data_object, sentence, seq_len):
    sentence = sentence.replace("-"," ")
    sentence = sentence.translate(None, string.punctuation).lower().split()
    seed_seq = data_object.get_sequence(sentence)
    res = []
    res.extend(sentence)
    vector_size = data_object.get_vector_size()
    for seq in seed_seq[:-1]:
        seq = seq.reshape((1,1,vector_size))
        predicted = lm_object.model_predict(seq)
    curr_input = seed_seq[-1].reshape((1,1,vector_size))
    for _ in xrange(seq_len):
        predicted = lm_object.model_predict(curr_input)[0]
        curr_word = data_object.get_word(predicted)
        res.append(curr_word)
        curr_input = data_object.get_vector(curr_word).reshape((1,1,vector_size))
    print " ".join(res)


if __name__=="__main__":
    lm_object, data_object = main("../data/glove.6B.50d.txt","../data/train.csv")
