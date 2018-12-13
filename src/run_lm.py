import sys
from lang_model_lstm import LSTMLanguageModel
import numpy as np
import string
import cPickle as pickle
import re
from DataGenerator import DataGenerator
from DataGeneratorSeq import DataGeneratorSeq
from DataGeneratorVec import DataGeneratorVector

def main(glove_fname, fname, vec=True, seq=False, vector_size=1):
    if vec:
        data_object = DataGeneratorVector()
    elif seq:
        data_object = DataGeneratorSeq(vector_size=vector_size)
    else:
        data_object = DataGenerator(vector_size=vector_size)
    lm_object = LSTMLanguageModel()
    infile = open(fname)
    first = True
    input_seq = []
    if vec:
        data_object.load_embeddings(glove_fname)
    for line in infile:
        if first:
            first = False
            continue
        text_id, text, author = line.split('","')
        if not author.startswith("EAP"): continue
        text = text.replace("-"," ")
        text = text.translate(None, string.punctuation).lower().split()
        text = [word for word in text if word.isalpha()]
        data_object.parse_text(text)
        if data_object.get_vocabulary_size() >= 5000:
            break
    if vec:
        lm_object.train_model_vec(data_object)
    else:
        lm_object.train_model(data_object)
    return lm_object, data_object

def generate_sentence(lm_object, data_object, sentence, seq_len, vec=True):
    sentence = sentence.replace("-"," ")
    sentence = sentence.translate(None, string.punctuation).lower().split()
    seed_seq = data_object.get_sequence(sentence)
    res = []
    res.extend(sentence)
    vector_size = data_object.get_vector_size()
    for seq in seed_seq[:-1]:
        if vec:
            seq = seq.reshape((1,1,vector_size))
        predicted = lm_object.model_predict(seq)

    if vec:
        curr_input = seed_seq[-1].reshape((1,1,vector_size))
    else:
        curr_input = seed_seq[-1].reshape((1,1))

    for _ in xrange(seq_len):
        predicted = lm_object.model_predict(curr_input)[0]
        curr_word = data_object.get_word(predicted)
        res.append(curr_word)
        if vec:
            curr_input = data_object.get_vector(curr_word).reshape((1,1,vector_size))
        else:
            curr_input = np.asarray([predicted]).reshape(1,1)

    print " ".join(res)


if __name__=="__main__":
    lm_object, data_object = main("../data/glove.6B.50d.txt","../data/train.csv")