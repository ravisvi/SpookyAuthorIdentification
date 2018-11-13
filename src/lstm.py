import utils
from constants import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential

def train_lstm(embedding_matrix, x_train, y_train, x_val, y_val, labels_index, word_index):
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                                EMBEDDING_DIM,
                                weights=[embedding_matrix],
                                input_length=MAX_SEQUENCE_LENGTH,
                                trainable=False))
    model.add(LSTM(64,return_sequences=True))
    model.add(Dropout(0.5))
    model.add(LSTM(64))
    model.add(Dense(len(labels_index), activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                optimizer='rmsprop',
                metrics=['acc'])

    model.fit(x_train, y_train, validation_data=(x_val, y_val),
            epochs=10, batch_size=128)
    return model