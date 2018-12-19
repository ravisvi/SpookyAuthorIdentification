import sys
from lang_model_lstm import LSTMLanguageModel
import numpy as np
import string
import cPickle as pickle
import re
from DataGenerator import DataGenerator
from DataGeneratorSeq import DataGeneratorSeq
from DataGeneratorVec import DataGeneratorVector

def clean_text(text):
    text = text.replace("-"," ")
    text = text.translate(None, string.punctuation).lower().split()
    text = [word for word in text if word.isalpha()]
    return text

def main(glove_fname, fname, vec=True, seq=False, vector_size=1, vocab_limit=10000, author_name="EAP"):
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
        if not author.startswith(author_name): continue
        text = clean_text(text)
        data_object.parse_text(text)
        if data_object.get_vocabulary_size() >= vocab_limit:
            break
    if vec:
        lm_object.train_model_vec(data_object)
    else:
        lm_object.train_model(data_object)
    return lm_object, data_object

def split_data(fname, test_limit=500):
    test_fname = "test2.csv"
    train_fname = "train2.csv"
    author_counts = {}
    infile = open(fname)
    first = True
    test_file = open(test_fname,"w")
    train_file = open(train_fname,"w")
    file_ptr = {}
    for line in infile:
        if first:
            train_file.write(line)
            first = False
            continue
        text_id, text, author = line.split('","')
        if not author in author_counts: 
            author_counts[author] = 0
            file_ptr[author] = test_file
        author_counts[author] += 1
        file_ptr[author].write(line)
        if author_counts[author] == test_limit:
            file_ptr[author] = train_file
    test_file.close()
    train_file.close()

def test_set_metrics(test_fname, lm_object, data_object, author_name="EAP"):
    infile = open(test_fname)
    test_matrix = np.asarray([])
    for line in infile:
        text_id, text, author = line.split('","') 
        if not author.startswith(author_name): continue
        text = clean_text(text)
        curr_test = data_object.get_sequence(text)
        if curr_test.shape[0] == 0: continue
        if test_matrix.shape[0] == 0:
            test_matrix = curr_test
        else:
            test_matrix = np.vstack((test_matrix, curr_test))
    test_seq = test_matrix[:-1]
    test_target = test_matrix[1:]
    print test_seq.shape, test_target.shape
    print lm_object.model_metrics(test_seq, test_target)

def generate_sentence(lm_object, data_object, sentence, seq_len, vec=True):
    sentence = sentence.replace("-"," ")
    sentence = sentence.translate(None, string.punctuation).lower().split()
    seed_seq = data_object.get_sequence(sentence)
    res = []
    res.extend(sentence)
    vector_size = data_object.get_vector_size()
    for seq in seed_seq[:-1]:
        predicted = lm_object.model_predict(seq)

    curr_input = seed_seq[-1].reshape((1,vector_size))

    for _ in xrange(seq_len):
        predicted = lm_object.model_predict(curr_input)[0]
        curr_word = data_object.get_word(predicted)
        res.append(curr_word)
        curr_input = np.asarray([predicted]).reshape((1,vector_size))

    print " ".join(res)


if __name__=="__main__":
    lm_object, data_object = main("../data/glove.6B.50d.txt","../data/train2.csv")
    #split_data("../data/train.csv")