import utils
import lstm
from datamodel import DataModel


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
    model_filename = 'models/lstm_model_2.json'
    weights_filename = 'models/lstm_model_2.h5'
    kaggle_filename = 'output/kaggle_lstm_128.csv'
    lstm.train_lstm(model_filename, weights_filename,
                    l1_d=128, l2_d=128, b_s=128)
    run_lstm_on_test_data(model_filename, weights_filename, kaggle_filename)


if __name__ == '__main__':
    main()
