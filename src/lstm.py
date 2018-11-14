import utils
from constants import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from datamodel import DataModel

def train_lstm(embedding_matrix, x_train, y_train, x_val, y_val, labels, word_index):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False))
    # 64
    model.add(LSTM(16,return_sequences=True))
    model.add(Dropout(0.5))
    # 64
    model.add(LSTM(16))
    model.add(Dense(labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])

    # 10, 128
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=2, batch_size=32)
    return model

def run_lstm():
    dm = DataModel()
    texts, labels = dm.get_train_data()
    word_index, data = utils.get_word_index(texts)
    x_train, y_train, x_val, y_val, _, _ = utils.get_train_val_test_data(data, labels)
    word_embeddings = utils.get_glove_embeddings()
    embedding_matrix = utils.get_embedding_matrix(word_embeddings, word_index)
    model = train_lstm(embedding_matrix, x_train, y_train, x_val, y_val, 3, word_index)
    return model