from nltk.corpus import stopwords
import string
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
from datamodel import DataModel
from word2veclite import Word2Vec
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
import constants

def clean_text(text):
    # remove punctuations
    cleaned_text = text.translate(None, string.punctuation)
    
    # upper to lower
    cleaned_text = [word.lower() for word in cleaned_text.split(' ')]

    # removing stopwords
    return ' '.join([word for word in cleaned_text if word not in stopwords.words('english')])

# Begin section Word2Vec utilites from github repository: https://github.com/cbellei/word2veclite
"""
Tokenize the corpus of text.
Args
    :param corpus: list containing a string of text (example: ["I like playing football with my friends"])
Returns:
    :return corpus_tokenized: indexed list of words in the corpus, in the same order as the original corpus (the example above would return [[1, 2, 3, 4]])
    :return V: size of vocabulary
"""
def tokenize(corpus):
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus_tokenized = tokenizer.texts_to_sequences(corpus)
    V = len(tokenizer.word_index)
    return corpus_tokenized, V

"""
Initialize the weights of the neural network.
Args
    :param V: size of the vocabulary
    :param N: size of the hidden layer
Returns:
    :return: weights W1, W2
"""
def initialize(V, N):
    np.random.seed(100)
    W1 = np.random.rand(V, N)
    W2 = np.random.rand(N, V)

    return W1, W2

"""Converts corpus text into context and center words
Args:
    corpus_tokenized: corpus text
    window_size: size of context window
Returns:
    context and center words (arrays)
"""
def corpus2io(corpus_tokenized, V, window_size):
    for words in corpus_tokenized:
        L = len(words)
        for index, word in enumerate(words):
            contexts = []
            center = []
            s = index - window_size
            e = index + window_size + 1
            contexts = contexts + [words[i]-1 for i in range(s, e) if 0 <= i < L and i != index]
            center.append(word-1)
            # x has shape c x V where c is size of contexts
            x = np_utils.to_categorical(contexts, V)
            # y has shape k x V where k is number of center words
            y = np_utils.to_categorical(center, V)
            yield (x, y)

# End section word2vec

def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum(axis=0)

"""
Generates word 2 vector embeddings.
"""
def generate_w2c_word_embeddings():
    dm = DataModel()
    train_x, _ = dm.get_train_data()
    
    w2c = Word2Vec(method="skipgram", corpus=list(train_x),
                window_size=5, n_hidden=128,
                n_epochs=3, learning_rate=0.08)
    
    W1, W2, loss_vs_epoch = w2c.run()

    pkl_dump = [W1, W2, loss_vs_epoch]
    with open('embeddings.pickle', 'wb') as handle:
        pickle.dump(pkl_dump, handle)

def get_w2c_word_embeddings():
    pkl_dump = None
    try:
        with open('embeddings.pickle', 'rb') as handle:
            pkl_dump = pickle.load(handle)
    except:
        print("Word embeddings not readable from the pickle file. Please generate them using the generate function.")
    if pkl_dump is not None:
        W1, W2, loss_vs_epoch = pkl_dump[0], pkl_dump[1], pkl_dump[2]
        return W1, W2, loss_vs_epoch
    else: 
        return None
"""
Get's the glove embeddings.
"""
def get_glove_embeddings():
    embeddings = {}
    f = open('glove.6B.100d.txt')
    for line in f:
        values = line.split()
        word = values[0]
        coefs = np.asarray(values[1:], dtype='float32')
        embeddings[word] = coefs
    f.close()
    return embeddings

"""
Get's the train, validate and test data.

Args:
    train_split: Ratio of data which will be training data. Remaining data will be split evenly to validate and test data.

Returns:
    x_train: Training data
    y_train: Training label
    x_val: Validation data
    y_val: Validation label
    x_test: Test data
    y_test: Test label
"""
def get_train_val_test_data(data, labels, train_split=0.8):
    train_size = len(data)
    indices = np.arange(data.shape[0])
    np.random.shuffle(indices)
    data = data[indices]
    labels = labels[indices]
    train_samples = int(train_split * train_size)

    val_samples = int(train_split * (train_size + (1-train_size)/2))

    x_train = data[:train_samples]
    y_train = labels[:train_samples]
    x_val = data[train_samples:val_samples]
    y_val = labels[train_samples:val_samples]
    x_test = data[val_samples:]
    y_test = labels[val_samples:]
    return x_train, y_train, x_val, y_val, x_test, y_test

"""
Get's the embedding matrix for each word in a list given embeddings.

Args:
    embeddings: Word2Vec embeddings or Glove embeddings etc.
    word_index: The words to which you need the matrix.

Returns:
    embedding_matrix: Embedding for each word in the list.
"""
def get_embedding_matrix(embeddings, word_index):
    embedding_matrix = np.zeros((len(word_index) + 1, constants.EMBEDDING_DIM))
    for word, i in word_index.items():
        embedding_vector = embeddings.get(word)
        if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
            embedding_matrix[i] = embedding_vector
    return embedding_matrix

"""
Get word index for the given text.

Args:
    texts: Getting the word index for the words in this text.
    max_unique_words: The max words to get the word index for. Default top 20000 words.

Returns:
    word_index: The word indices for the embeddings of each unique word in the texts.
    data: pad sequenced data for the input texts
"""
def get_word_index(texts, max_unique_words=20000):

    tokenizer = Tokenizer(nb_words=max_unique_words)
    tokenizer.fit_on_texts(texts)
    sequences = tokenizer.texts_to_sequences(texts)

    word_index = tokenizer.word_index
    print('Found %s unique tokens.' % len(word_index))
    data = pad_sequences(sequences, maxlen=constants.MAX_SEQUENCE_LENGTH)
    return word_index, data
