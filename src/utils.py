from nltk.corpus import stopwords
import string
from keras.utils import np_utils
from keras.preprocessing.text import Tokenizer
import numpy as np
import pickle
from datamodel import DataModel
from word2veclite import Word2Vec

def clean_text(text):
    # remove punctuations
    cleaned_text = text.translate(None, string.punctuation)
    
    # upper to lower
    cleaned_text = [word.lower() for word in cleaned_text.split(' ')]

    # removing stopwords
    return ' '.join([word for word in cleaned_text if word not in stopwords.words('english')])

# Begin section Word2Vec utilites from github repository: https://github.com/cbellei/word2veclite
def tokenize(corpus):
    """
    Tokenize the corpus of text.
    :param corpus: list containing a string of text (example: ["I like playing football with my friends"])
    :return corpus_tokenized: indexed list of words in the corpus, in the same order as the original corpus (the example above would return [[1, 2, 3, 4]])
    :return V: size of vocabulary
    """
    tokenizer = Tokenizer()
    tokenizer.fit_on_texts(corpus)
    corpus_tokenized = tokenizer.texts_to_sequences(corpus)
    V = len(tokenizer.word_index)
    return corpus_tokenized, V

def initialize(V, N):
    """
    Initialize the weights of the neural network.
    :param V: size of the vocabulary
    :param N: size of the hidden layer
    :return: weights W1, W2
    """
    np.random.seed(100)
    W1 = np.random.rand(V, N)
    W2 = np.random.rand(N, V)

    return W1, W2

def corpus2io(corpus_tokenized, V, window_size):
    """Converts corpus text into context and center words
    # Arguments
        corpus_tokenized: corpus text
        window_size: size of context window
    # Returns
        context and center words (arrays)
    """
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

def generate_word_embeddings():
    dm = DataModel()
    train_x, _ = dm.get_train_data()
    
    w2c = Word2Vec(method="skipgram", corpus=list(train_x),
                window_size=7, n_hidden=2,
                n_epochs=10, learning_rate=0.8)
    
    W1, W2, loss_vs_epoch = w2c.run()

    pkl_dump = [W1, W2, loss_vs_epoch]
    with open('embeddings.pickle', 'wb') as handle:
        pickle.dump(pkl_dump, handle)

def get_word_embeddings():
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