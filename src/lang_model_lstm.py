from keras.preprocessing.sequence import pad_sequences
from keras.layers import Embedding, LSTM, Dense, Dropout, TimeDistributed, Activation
from keras.preprocessing.text import Tokenizer
from keras.callbacks import EarlyStopping, ModelCheckpoint
from keras.models import Sequential
import keras.utils as ku 
import numpy as np 

class LSTMLanguageModel:
    def __init__(self, embedding_size=256, layer_size=256):
        self.tokenizer = Tokenizer()
        self.reverse_index = {}
        self.max_len = -1*float("inf")
        self.embedding_size = embedding_size
        self.layer_size = layer_size

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

    def train_model_generator(self, data_generator, data_path=""):
        self.model = Sequential()
        vocabulary = data_generator.vocabulary
        num_steps = data_generator.num_steps
        batch_size = data_generator.batch_size
        self.model.add(Embedding(vocabulary, self.embedding_size, input_length=num_steps))
        self.model.add(LSTM(self.layer_size, return_sequences=True))
        self.model.add(LSTM(self.layer_size, return_sequences=True))
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
        label = ku.to_categorical(label, num_classes=total_words)
        self.model = Sequential()
        self.model.add(Embedding(total_words, self.embedding_size, input_length=predictors.shape[1]))
        self.model.add(LSTM(self.layer_size))#, return_sequences=True))
        self.model.add(Dropout(0.3))
        self.model.add(Dense(total_words, activation='softmax'))

        self.model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
        earlystop = EarlyStopping(monitor='val_loss', min_delta=0, patience=5, verbose=0, mode='auto')
        self.model.fit(predictors, label, epochs=50, verbose=1, callbacks=[earlystop], batch_size=512)
        print(self.model.summary())


    def generate_text(self, seed_text, next_words, max_sequence_len):
        for _ in range(next_words):
            token_list = self.tokenizer.texts_to_sequences([seed_text])[0]
            token_list = pad_sequences([token_list], maxlen=self.max_len-1, padding='pre')
            predicted = self.model.predict_classes(token_list, verbose=0)
            
            output_word = self.reverse_index[predicted]
            seed_text += " " + output_word
        return seed_text
