##############################################################################
###                              dataLoader.py                             ###
##############################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class dataGenerator(object):
    def __init__(self, data, lb, train_val_split=0.8, test_windows=30):
        self.data = np.array(data)
        self.lb = lb
        self.train_val_split = train_val_split
        self.test_windows=test_windows
        self.__instance = self._get_train_instance(self.data[:-self.test_windows], self.lb)
        self.__train, self.__val = self._train_val_split(X=self.__instance[0], y=self.__instance[1], split=self.train_val_split)
    
    def get_train_data(self):
        return self.__train
    
    def get_val_data(self):
        return self.__val
    
    def get_test_data(self):
        X = np.reshape(self.data[-self.lb-self.test_windows:-self.test_windows], (1,-1,1))
        y = np.array(self.data[-self.test_windows:])
        return X, y
    
    def _get_train_instance(self, data, lb):
        '''
        Generate training instances from series data.
        '''
        X, y = [], []
        for i in range(len(data) - lb):
            X.append(np.reshape(data[i:(i+lb)], (-1,1)))
            y.append(data[i+lb])
        return np.array(X), np.array(y)
    
    def _train_val_split(self, X, y, split):
        '''
        Split training and validation data.
        '''
        X_train, X_val = X[:int(len(X)*split)], X[int(len(X)*split):]
        y_train, y_val = y[:int(len(y)*split)], y[int(len(y)*split):]
        return (X_train, y_train), (X_val, y_val)

        