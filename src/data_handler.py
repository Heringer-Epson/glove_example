import numpy as np

class Data_Handler(object):
    """
    Description:
    ------------
    Collection of functions to treat data for NLP applications.
    Based on functions originally written by Jason Brownlee.
    Source: https://machinelearningmastery.com/develop-word-embedding-model-predicting-movie-review-sentiment/

    Parameters:
    -----------
    None needed.

    Return:
    -------
    According to function usage.
    """        
    def __init__(self):
        pass

    def load_embedding(self, fname):
        '''Function to load and parse the pre-trained word embedding for GloVe.
        '''
        #Load content to a list.
        with open(fname) as f: 
            lines = f.readlines()

        #Parse the data
        embedding = dict()
        for line in lines:
            word_coef = line.split()
            embedding[word_coef[0]] = np.asarray(word_coef[1:], dtype='float32')
        return embedding
        
    # create a weight matrix for the Embedding layer from a loaded embedding
    def get_weight_matrix(self, embedding, vocab):
        vocab_size = len(vocab) + 1 #+1 account for unknown words.
        coef_matrix = np.zeros((vocab_size, 100)) #Initialize matrix.

        #For words present in the local corpus, retrieve coeffs from the
        #pre-trained GloVe model.
        for word, i in vocab.items():
            vector = embedding.get(word)
            if vector is not None:
                coef_matrix[i] = vector
        return coef_matrix
