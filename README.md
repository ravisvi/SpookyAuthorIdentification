# SpookyAuthorIdentification

Download the `train.csv` file from here:
https://www.kaggle.com/c/spooky-author-identification/

The main script to run this program is `run_lm.py`. Before, we start to train our model, let's first set-up our training and test data. Use the function `split_data` to split the training file dowloaded from Kaggle (train.csv) into train and test files. The function takes two arguments: the path to the training file (which is a required argument) and `test_limit` which specifies how many test samples you want to create per author (optional, default `test_limit` is 5000)

```bash
>> from run_lm import *
>> split_data("../data/train.csv",test_limit=250)
```

This will create two files `train2.csv` and `test2.csv`, which we'll use for further training.

Now to build the model, we use the function `main` in `run_lm.py`. This function takes two required parameters: the location to the file containing glove embeddings and the location to the file containing training sample. Other optional samples are `vocab_limit` which limits the vocabulary size to this number (set to 10000 by default), and `author_name` which should be one of `"EAP"`,`"HPL"` and `"MWS"` (default set to `"EAP"`)

```bash
>> glove_fname = "../data/glove.6B.50d.txt"
>> train_fname = "../data/train2.csv"
>> lm_object, data_object = main(glove_fname, train_fname, vocab_limit=20000, author_name="MWS")
```

The function `main` returns two objects: `lm_object`, an object of the language model and `data_object`, an object of the data class. Now to generate text from language model, use the function `generate_sentence` and provide lm_object, data_object, a seed text and the number of words to be generated as required arguments

```bash
>> generate_sentence(lm_object, data_object, "the man sat on the", 4)
```

Keep in mind that the seed text given should only contain words seen in the vocabulary before, otherwise the function will throw an exception.

To calculate perplexity and accuracy on the test set, use the function `test_set_metrics` which takes the path to the test file, `lm_object` and `data_object` as required arguments. You can also specify `author_name` as optional argument (default is set to `EAP`):

```bash
>> test_fname = "../data/test2.csv"
>> test_set_metrics(test_fname, lm_object, data_object, author_fname="HPL")
```