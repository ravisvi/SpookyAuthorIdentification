import utils

def main():     
    utils.generate_word_embeddings()

    W1, W2, loss_vs_epochs = utils.get_word_embeddings()
    
    print('W1: ' + str(W1))
    print('W2: ' + str(W2))
    print('Loss vs Epochs' + str(loss_vs_epochs))

if __name__ == "__main__":
    main()