from flair.data import Corpus
from flair.datasets import ColumnCorpus

class NERCorpus:
    def __init__(self, data_folder) -> None:
        self.data_folder = data_folder
        
    def create_corpus(self) -> Corpus:
        corpus = ColumnCorpus(data_folder = '{}'.format(self.data_folder), 
                            column_format = {0: 'text', 1: 'ner'},
                            train_file = 'train.iob2',
                            test_file = 'test.iob2',
                            dev_file = 'valid.iob2')
        return corpus