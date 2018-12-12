import utils
from constants import MAX_SEQUENCE_LENGTH, EMBEDDING_DIM
from keras.layers import Dense, Dropout, Embedding, LSTM, Bidirectional
from keras.models import Sequential
from datamodel import DataModel


def _lstm(embedding_matrix,
          x_train,
          y_train,
          x_val,
          y_val,
          labels,
          word_index,
          l1_d,
          l2_d,
          b_s):
    """
    Trains a 2 layer lstm and returns the model fit on the input data.

    Args:
        embedding_matrix: The embedding matrix for the input
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        labels: Total number of unique labels
        word_index: Word Index map according to the embedding used
        l1_d: Layer one dimensions
        l2_d: Layer two dimensions
        b_s: Batch size

    Returns:
        model: The lstm model fit on the training data
    """
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))
    # 64
    model.add(LSTM(l1_d, return_sequences=True))
    model.add(Dropout(0.3))
    # 64
    model.add(LSTM(l2_d))
    model.add(Dense(labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # 10, 128
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=100, batch_size=b_s)
    return model


def _bi_lstm(embedding_matrix,
          x_train,
          y_train,
          x_val,
          y_val,
          labels,
          word_index,
          l1_d,
          l2_d,
          b_s):
    """
    Trains a 2 layer lstm and returns the model fit on the input data.

    Args:
        embedding_matrix: The embedding matrix for the input
        x_train: Training data
        y_train: Training labels
        x_val: Validation data
        y_val: Validation labels
        labels: Total number of unique labels
        word_index: Word Index map according to the embedding used
        l1_d: Layer one dimensions
        l2_d: Layer two dimensions
        b_s: Batch size

    Returns:
        model: The lstm model fit on the training data
    """
    model = Sequential()
    model.add(Embedding(len(word_index) + 1,
                        EMBEDDING_DIM,
                        weights=[embedding_matrix],
                        input_length=MAX_SEQUENCE_LENGTH,
                        trainable=False))
    # 64
    model.add(Bidirectional(LSTM(l1_d, return_sequences=True)))
    model.add(Dropout(0.3))
    # 64
    model.add(Bidirectional(LSTM(l2_d)))
    model.add(Dense(labels, activation='softmax'))
    model.compile(loss='categorical_crossentropy',
                  optimizer='rmsprop',
                  metrics=['acc'])

    # 10, 128
    model.fit(x_train, y_train, validation_data=(x_val, y_val),
              epochs=100, batch_size=b_s)
    return model


def train_lstm(model_filename, weights_filename, l1_d=128, l2_d=128, b_s=128, bi=False):
    """
    Trains a 2 layer lstm and saves the model in the files specified.

    Args:
        model_filename: The filename to save the model
        weights_filename: The filename to save the weights
        l1_d: Layer one dimensions
        l2_d: Layer two dimensions
        b_s: Batch size

    Returns:
        model: The lstm model fit on the training data
    """
    dm=DataModel()
    texts, labels=dm.get_train_data()
    word_index, data=utils.get_word_index(texts)
    x_train, y_train, x_val, y_val, _, _=utils.get_train_val_test_data(
        data, labels)
    word_embeddings=utils.get_glove_embeddings()
    embedding_matrix=utils.get_embedding_matrix(word_embeddings, word_index)
    if bi:
        model=_bi_lstm(embedding_matrix, x_train, y_train, x_val,
                  y_val, 3, word_index, l1_d, l2_d, b_s)
    else:
        model=_lstm(embedding_matrix, x_train, y_train, x_val,
                  y_val, 3, word_index, l1_d, l2_d, b_s)
    save_model(model, model_filename, weights_filename)
    return model


def save_model(model, model_filename, weights_filename):
    """
    Trains a 2 layer lstm and returns the model fit on the input data.

    Args:
        model: The lstm model fit on the training data
        model_filename: The filename to save the model
        weights_filename: The filename to save the weights
    """
    # serialize model to JSON
    model_json=model.to_json()
    with open(model_filename, "w") as json_file:
        json_file.write(model_json)

    # serialize weights to HDF5
    model.save_weights(weights_filename)
    print("Saved model to disk")
