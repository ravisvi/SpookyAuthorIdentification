import utils
import lstm
from datamodel import DataModel
from keras.utils import to_categorical
import numpy as np


def w2c():
    utils.generate_w2c_word_embeddings()

    W1, W2, loss_vs_epochs = utils.get_w2c_word_embeddings()

    print('W1: ' + str(W1))
    print('W2: ' + str(W2))
    print('Loss vs Epochs' + str(loss_vs_epochs))


def run_lstm_on_test_data(model_filename, weights_filename, kaggle_filename):
    dm = DataModel()
    test_x, test_id = dm.get_test_data()
    print(test_x.shape)
    _, data = utils.get_word_index(test_x)
    # word_embeddings = utils.get_glove_embeddings()
    # _ = utils.get_embedding_matrix(word_embeddings, word_index)
    y = utils.load_model_and_evaluate(model_filename, weights_filename, data)
    utils.write_kaggle_file(y, test_id, kaggle_filename)


def main():
    model_filename = 'models/lstm_model_3.json'
    weights_filename = 'models/lstm_model_3.h5'
    kaggle_filename = 'output/kaggle_lstm_256.csv'
    lstm.train_lstm(model_filename, weights_filename,
                    l1_d=256, l2_d=256, b_s=128)
    run_lstm_on_test_data(model_filename, weights_filename, kaggle_filename)

    model_filename = 'models/bi_lstm_model_128.json'
    weights_filename = 'models/bi_lstm_model_128.h5'
    kaggle_filename = 'output/bi_kaggle_lstm_128.csv'
    lstm.train_lstm(model_filename, weights_filename,
                    l1_d=128, l2_d=128, b_s=128, bi=True)
    run_lstm_on_test_data(model_filename, weights_filename, kaggle_filename)

    model_filename = 'models/bi_lstm_model_265.json'
    weights_filename = 'models/bi_lstm_model_256.h5'
    kaggle_filename = 'output/bi_kaggle_lstm_256.csv'
    lstm.train_lstm(model_filename, weights_filename,
                    l1_d=256, l2_d=256, b_s=128, bi=True)
    run_lstm_on_test_data(model_filename, weights_filename, kaggle_filename)

def accuracy_test(model_filename, weights_filename):
    dm = DataModel()
    x_train, y_train = dm.get_train_data()
    train_samples = int(0.8 * len(x_train))
    x_test = x_train[train_samples:]
    y_test = y_train[train_samples:]

    _, data = utils.get_word_index(x_test)
    labels = to_categorical(np.asarray(y_test))
    # word_embeddings = utils.get_glove_embeddings()
    # _ = utils.get_embedding_matrix(word_embeddings, word_index)
    md = utils.load_model(model_filename, weights_filename)
    result = md.evaluate(data, labels)
    print('\nTest loss:', result[0])
    print('Test accuracy:', result[1])

if __name__ == '__main__':
    main()
    accuracy_test('models/lstm_model_3.json', 'models/lstm_model_3.h5')
    # accuracy_test('models/lstm_model.json', 'models/lstm_model.h5')
    # accuracy_test('models/lstm_model_2.json', 'models/lstm_model_2.h5')
