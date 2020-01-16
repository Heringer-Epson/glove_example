import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score

from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Flatten
from keras.layers import Embedding

from data_handler import Data_Handler

class Glove_Experiment(object):
    """
    Description:
    ------------
    Uses Glove for text embedding on a labeled corpus. Then applies an ANN to
    predict labels on test data. 

    Parameters:
    -----------
    embedding_dim : ~int
        Dimension of vectors in the space in which words are represented.
        Accepts 100, 200 or 300. Default is 100.

    Source:
    -------
    The CNN model was adapt from 
    https://www.kaggle.com/francoisdubois/build-a-word-embedding-with-glove-matrix
    
    Return:
    -------
    None
    """        
    def __init__(self, embedding_dim=100):
        self.embedding_dim = embedding_dim
        self.seed = 234012587
        self.test_split_fraction = .2
        
        self.df = None
        self.tokenizer = None
        self.Train_X, self.Test_X = None, None
        self.Train_y, self.Test_y = None, None
        self.max_length, self.vocab_size = None, None
        self.embedding_vectors = None
        self.model = None
        
        self.run_experiment()

    @property
    def embedding_dim(self):
        return self._embedding_dim
    
    @embedding_dim.setter
    def embedding_dim(self, value):
        if value not in [100, 200, 300]:
            raise ValueError('Embedding dimension must be 100, 200 or 300.')
        self._embedding_dim = value
        
    def collect_data(self):
        self.df = pd.read_csv('./../data/corpus.csv', encoding='latin-1')
        self.df.dropna(how='all', inplace=True)        

    def split_data(self):
        X = self.df.text
        y = self.df.label
        self.Train_X, self.Test_X, self.Train_y, self.Test_y = train_test_split(
          X, y, test_size=self.test_split_fraction, random_state=self.seed)

        self.max_length = max([len(s.split()) for s in self.Train_X])

    def encode_target(self):
        Encoder = LabelEncoder()
        self.Train_y = Encoder.fit_transform(self.Train_y)
        self.Test_y = Encoder.fit_transform(self.Test_y)        

    def tokenize_data(self):
        self.tokenizer = Tokenizer()
        self.tokenizer.fit_on_texts(self.Train_X)
        
        self.vocab_size = len(self.tokenizer.word_index) + 1

    def encode_data(self):
        encoded_docs = self.tokenizer.texts_to_sequences(self.Train_X)

        #Train sample.
        self.Train_X = pad_sequences(
          encoded_docs, maxlen=self.max_length, padding='post')

        #Test sample.
        encoded_docs = self.tokenizer.texts_to_sequences(self.Test_X)
        self.Test_X = pad_sequences(
          encoded_docs, maxlen=self.max_length, padding='post')

    def perform_embedding(self):
        
        DH = Data_Handler()
        
        #Parse pre-trained embedding.
        raw_embedding = DH.load_embedding(
          './../data/glove.6B.%dd.txt' %self._embedding_dim)

        #Create matrix with embedding for the words present in the corpus.
        self.embedding_vectors = DH.get_weight_matrix(
          raw_embedding, self.tokenizer.word_index)
        
    def create_neural_model(self):

        self.model = Sequential()
        self.model.add(Embedding(
                       len(self.tokenizer.word_index) + 1,
                       self.embedding_dim, input_length=self.max_length))
        self.model.add(Flatten())
        self.model.add(Dense(32, activation='relu'))
        self.model.add(Dense(1, activation='sigmoid'))
        self.model.summary()

        self.model.layers[0].set_weights([self.embedding_vectors])        

    def fit_model(self):
        self.model.compile(
          loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        self.model.fit(self.Train_X, self.Train_y, epochs=5, verbose=2)
        loss, acc = self.model.evaluate(self.Test_X, self.Test_y, verbose=0)
        print('Test Accuracy: %f' % (acc*100))

    def run_experiment(self):
        self.collect_data()
        self.split_data()
        self.encode_target()
        self.tokenize_data()
        self.encode_data()
        self.perform_embedding()
        self.create_neural_model()
        self.fit_model()

if __name__ == '__main__':
    Glove_Experiment()
