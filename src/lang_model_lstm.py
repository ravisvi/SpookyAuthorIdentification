from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Activation, Flatten
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku 
import numpy as np
import string 
import keras.backend as K

class LSTMLanguageModel:
    def __init__(self, embedding_size=50, layer_size=128, batch_size=128, epochs=100):
        self.tokenizer = Tokenizer()
        self.reverse_index = {}
        self.max_len = -1*float("inf")
        self.embedding_size = embedding_size
        self.layer_size = layer_size
        self.batch_size = batch_size
        self.epochs = epochs

    def perplexity(self, y_true, y_pred):
        print y_true, y_pred
        cross_entropy = K.sparse_categorical_crossentropy(y_true, y_pred)
        perplexity = K.exp(cross_entropy)
        return perplexity

    def train_model_generator(self, data_generator, data_path=""):
        self.model = Sequential()
        vocabulary = data_generator.vocabulary
        num_steps = data_generator.num_steps
        batch_size = data_generator.batch_size
        self.model.add(Embedding(vocabulary, self.embedding_size, input_length=num_steps))
        self.model.add(LSTM(self.layer_size, return_sequences=True))
        #self.model.add(LSTM(self.layer_size, return_sequences=True))
        self.model.add(Dropout(0.5))
        self.model.add(TimeDistributed(Dense(vocabulary)))
        self.model.add(Activation('softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['categorical_accuracy'])
        checkpointer = ModelCheckpoint(filepath=data_path, verbose=1)
        num_epochs = 50
        model.fit_generator(data_generator.generate(), len(train_data)//(batch_size*num_steps), num_epochs, callbacks=[checkpointer])
        print(self.model.summary())

    def train_model(self, data_object):
        predictors, label = data_object.get_inputs()
        total_words = data_object.get_vocabulary_size()
        print total_words
        self.vector_size = predictors.shape[1]
        #label = ku.to_categorical(label, num_classes=total_words)
        print predictors.shape, label.shape
        self.model = Sequential()
        self.model.add(Embedding(total_words, self.embedding_size, input_length=self.vector_size))
        self.model.add(LSTM(self.layer_size, return_sequences=True))
        self.model.add(LSTM(self.layer_size))
        self.model.add(Dense(self.layer_size, activation='relu'))
        self.model.add(Dense(total_words, activation='softmax'))
        print(self.model.summary())

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', self.perplexity])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.model.fit(predictors, label, epochs=self.epochs, verbose=1, callbacks=[earlystop], batch_size=self.batch_size)

    def train_model_vec(self, data_object):
        predictors, label = data_object.get_inputs()
        total_words = data_object.get_vocabulary_size()
        print total_words
        self.vector_size = predictors.shape[2]
        label = ku.to_categorical(label, num_classes=total_words)
        self.model = Sequential()
        self.model.add(LSTM(self.layer_size, input_shape=(1, self.vector_size), return_sequences=True))
        self.model.add(LSTM(self.layer_size))
        self.model.add(Dense(self.layer_size, activation='relu'))
        self.model.add(Dense(total_words, activation='softmax'))
        print(self.model.summary())

        self.model.compile(loss='sparse_categorical_crossentropy', optimizer='adam', metrics=['accuracy', self.perplexity])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.model.fit(predictors[:1], label[:1], epochs=self.epochs, verbose=1, callbacks=[earlystop], batch_size=self.batch_size)

    def model_predict(self, input_seq):
        return self.model.predict_classes(input_seq, verbose = 0).tolist()

