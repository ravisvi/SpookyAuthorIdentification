import pandas as pd

class DataModel:
    def __init__(self):
        self._train_df = pd.read_csv("data/train.csv")
        self._test_df = pd.read_csv("data/test.csv")
        self.author_mapping_dict = {'EAP':0, 'HPL':1, 'MWS':2}
        self.train_x = self._train_df['text'].values
        self.train_y = self._train_df['author'].map(self.author_mapping_dict)
        self.train_id = self._train_df['id'].values
        self.test_id = self._test_df['id'].values
        self.test_x = self._test_df['text'].values
        # self.test_y = self._test_df['author'].map(self.author_mapping_dict)

    def get_train_data(self):
        return self.train_x, self.train_y

    def get_test_data(self):
        pass
        # return self.test_x, self.test_y
        
    def reinit_data(self):
        self._train_df = pd.read_csv("data/train.csv")
        self._test_df = pd.read_csv("data/test.csv")