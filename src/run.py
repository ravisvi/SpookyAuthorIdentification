import utils
import lstm

def w2c():
    utils.generate_w2c_word_embeddings()

    W1, W2, loss_vs_epochs = utils.get_w2c_word_embeddings()
    
    print('W1: ' + str(W1))
    print('W2: ' + str(W2))
    print('Loss vs Epochs' + str(loss_vs_epochs))

def main():     
   lstm.run_lstm()

if __name__ == "__main__":
    main()