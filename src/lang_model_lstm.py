from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping
from keras.models import Sequential
import keras.utils as ku 
import numpy as np 

class LSTMLanguageModel:
    def __init__(self):
        self.tokenizer = Tokenizer()
        self.reverse_index = {}
        self.max_len = -1*float("inf")

    def text_to_seq(self, text):
        self.tokenizer.fit_on_texts([text])
        token_list = self.tokenizer.texts_to_sequences([text])[0]
        input_sequences = []
        for i in range(1, len(token_list)):
            sequence = token_list[:i+1]
            if len(sequence) > self.max_len: self.max_len = len(sequence)
            input_sequences.append(sequence)
        return input_sequences

    def get_input(self, input_sequences):
        return np.array(pad_sequences(input_sequences, maxlen=self.max_len, padding='pre'))

    def finish_corpus(self):
        self.total_words = len(self.tokenizer.word_index) + 1

    def train_model(self, input_sequences):
        predictors, label = input_sequences[:,:-1],input_sequences[:,-1]
        label = ku.to_categorical(label, num_classes=self.total_words)
        self.model = Sequential()
        self.model.add(Embedding(self.total_words, 10, input_length=self.max_len - 1))
        self.model.add(LSTM(150, return_sequences=True))
        self.model.add(Dropout(0.2))
        self.model.add(LSTM(100))
        self.model.add(Dense(self.total_words, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.model.fit(predictors, label, epochs=1, verbose=1, callbacks=[earlystop], batch_size=512)
        print(self.model.summary())

    def generate_reverse_index(self):
        for word, index in self.tokenizer.word_index.items():
            self.reverse_index[index] = word

    def generate_text(self, seed_text, next_words, max_sequence_len):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_len-1, padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=0)
            
            output_word = self.reverse_index[predicted]
            seed_text += " " + output_word
        return seed_text
