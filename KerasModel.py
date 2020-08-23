#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import numpy as np
import os
from sklearn.base import BaseEstimator, ClassifierMixin, TransformerMixin
from keras import backend as K
from sklearn.utils.class_weight import compute_class_weight
from keras.models import Sequential
from keras.layers import Dense, Activation, Embedding, Dropout, LSTM, SpatialDropout1D
from keras.layers.convolutional import Conv1D
from keras.layers.embeddings import Embedding
from keras.models import Model
from keras.layers import Input


# In[ ]:


class WeightedMetrics(object):
    def __init__(self):
        pass
    def f_1_score_weighted(self, y_true, y_pred, weights):
        pred_labels = np.asarray(y_pred)
        true_labels = np.asarray(y_true)
        weights = np.asarray(weights)
        TP = np.sum(weights[np.logical_and(pred_labels == 1, true_labels == 1)])
        TN = np.sum(weights[np.logical_and(pred_labels == 0, true_labels == 0)])
        FP = np.sum(weights[np.logical_and(pred_labels == 1, true_labels == 0)])
        FN = np.sum(weights[np.logical_and(pred_labels == 0, true_labels == 1)])
        def precision_weighted(TP, TN, FP):
            return TP/(TP+FP)
        def recall_weighted(TP, TN, FN):
            return TP/(TP+FN)
        F_1 = 2 * (precision_weighted(TP, TN, FP) * recall_weighted(TP, TN, FN)) / (precision_weighted(TP, TN, FP) + 
                                                                                    recall_weighted(TP, TN, FN))
        return F_1
    def integrated_crit_weighted(self, data_true, data_pred, PATH):
        #load weights
        weights_criterias = np.load(os.path.join(PATH, 'weights_criterias.npy')).item()
        n = 0
        sums = 0
        for column in data_pred:
            sums+=self.f_1_score_weighted(data_true[column].astype(int), data_pred[column].astype(int), 
                                          weights_criterias[column])
            n+=weights_criterias[column]
        crit_new = sums/n
        return crit_new  


# In[ ]:


class ModelLSTM(BaseEstimator, TransformerMixin, ClassifierMixin):
    def __init__(self, dict_params, data_criterias, embedding=None):
        self.model_params = dict_params
        self.data_criterias = data_criterias
        self.embedding = embedding
        
    def calculating_class_weights(self, y):
        number_dim = np.shape(y)[1]
        weights = np.empty([number_dim, 2])
        for i in range(number_dim):
            if len(np.unique(y.values[:, i])) == 1:
                weights[i] = compute_class_weight(None, np.unique(y.values[:, i]), y.values[:, i])
            else:
                weights[i] = compute_class_weight('balanced', [0.,1.], y.values[:, i])
        return weights 
    
    def get_weighted_loss(self, weights):
        def weighted_loss(y_t, y_pred):
            return K.mean((weights[:,0]**(1-y_t))*(weights[:,1]**(y_t))*K.binary_crossentropy(y_t, y_pred), axis=-1)
        return weighted_loss
    
    def create_model(self, sigmoid_shape, one_hot_shape):
        main_input = Input(shape=(self.model_params['max_word'], ))
        # This embedding layer will encode the input sequence
        x = Embedding(self.model_params['vocab_size']+1, self.model_params['embed_len'], 
                      input_length=self.model_params['max_word'], weights=[self.embedding])(main_input)
        
        x = SpatialDropout1D(0.4)(x)
        x = Conv1D(self.model_params['max_word'], 5, padding = 'same', activation='relu')(x)
        x = MaxPooling1D(pool_size=2)(x)
        lstm_out = LSTM(self.model_params['neurons'])(x)
        
        # add one-hot input layer
        auxiliary_input = Input(shape=(one_hot_shape, ))
        x = keras.layers.concatenate([lstm_out, auxiliary_input])
        # And finally we add the main logistic regression layer
        main_output = Dense(sigmoid_shape, activation='sigmoid')(x)

        model = Model(inputs=[main_input, auxiliary_input], outputs=main_output)
        return model
    
    def split_data(self, data):
        self.num_classes = self.data_criterias.shape[1]
        self.one_hot_shape = data[:, self.model_params['max_word']:].shape[1]
        model = self.create_model(self.num_classes, self.one_hot_shape)
        #split on train and validation data
        x_train_shuf, x_test_shuf, y_train_shuf,        y_test_shuf, weights_train, weights_test = train_test_split(data, self.data_criterias, self.weights,
                                                                                      test_size = 0.2, random_state = 0)
        return model, x_train_shuf, x_test_shuf, y_train_shuf, y_test_shuf, weights_train, weights_test

