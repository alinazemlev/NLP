#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np 
import pandas as pd
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
import os
import keras
from keras.preprocessing.text import Tokenizer
import pickle
import re
from keras.preprocessing.sequence import pad_sequences


# In[ ]:


class Tokenize(BaseEstimator, TransformerMixin):
    def __init__(self, dict_params, use_vocab = True, save_tok = True):
        self.vocab_size = dict_params['vocab_size'] 
        self.maxlen = dict_params['max_word']
        self.use_vocab = use_vocab
        self.save_tok = save_tok
    def fit(self, X):
        self.tokenizer = Tokenizer(num_words=self.vocab_size+1)
        self.tokenizer.fit_on_texts(X)
        return self
    def transform(self, X):
        X_toc = self.tokenizer.texts_to_sequences(X)
        X_toc = pad_sequences(X_toc, maxlen=self.maxlen)
        if self.save_tok:
            with open('tokenizer.pickle', 'wb') as handle:
                pickle.dump(self.tokenizer, handle, protocol=pickle.HIGHEST_PROTOCOL)
        if self.use_vocab:
            index2word = self.tokenizer.index_word
            index2word_new = {index2word[k]:k for k in range(1, self.vocab_size+1)}
            return index2word_new 
        else:
            return X_toc          


# In[ ]:


class BuildEmbedding(BaseEstimator, TransformerMixin):
    def __init__(self, w2v, dict_params):
        self.w2v = w2v
        self.vocab_size = dict_params['vocab_size']
    def fit(self, vocab):
        return self
    def transform(self, vocab):
        embedding_matrix = np.zeros((self.vocab_size+1, 300))
        for word, i in vocab.items():
            embedding_vector =self.w2v.get(word)
            if embedding_vector is not None:
            # words not found in embedding index will be all-zeros.
                embedding_matrix[i] = embedding_vector
        #np.save('embedding_matrix_rus.npy', embedding_matrix)
        return embedding_matrix


