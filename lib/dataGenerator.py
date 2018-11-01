##############################################################################
###                              dataLoader.py                             ###
##############################################################################

import numpy as np
import pandas as pd
from sklearn.preprocessing import MinMaxScaler

class dataGenerator(object):
    def __init__(self, data, lb, lf, normalize_window=None, train_val_split=0.8, test_windows=30):
        '''
        == parameters ==:
        data: time series data
        lb: number of time steps to look back for forecast
        lf: number of time steps to forecast
        normalize_window: whether normalize each window indepedently
        train_val_split: percentage for training data
        test_window: time steps to forecast for testing
        '''
        self.data = np.array(data)
        self.lb = lb
        self.lf = lf
        self.normalize_window = normalize_window
        self.train_val_split = train_val_split
        self.test_windows = test_windows
        self.__instance = self._get_train_instance(data=self.data[:-self.test_windows])
        self.__train, self.__val = self._train_val_split(X=self.__instance[0], 
                                                         y=self.__instance[1], 
                                                         base=self.__instance[2])
    
    def get_train_data(self):
        return self.__train
    
    def get_val_data(self):
        return self.__val
    
    def get_test_data(self):
        X, y, base = self._get_window_data(self.data[-self.lb-self.test_windows:])
        X = np.reshape(X, (1,-1,1))
        return np.reshape(X, (1,-1,1)), np.reshape(y, (-1,1)), np.reshape(base, (-1,1))
    
    def inverse_transform(self, y, base):
        '''
        Inverse transform normalized data to original scale.
        '''
        if self.normalize_window == 'ratio':
            y = y * base
        elif self.normalize_window == 'percent_change':
            y = (y / 100 + 1) * base
        elif self.normalize_window is None:
            pass
        else:
            raise ValueError('Unrecognizable normalization method.')
        return y
    
    def build_prediction_table()
    
    def _get_train_instance(self, data):
        '''
        Generate training instances from series data.
        '''
        lb, lf = self.lb, self.lf
        X, y, base = [], [], []
        for i in range(len(data) - lb - lf + 1):
            tmp_X, tmp_y, tmp_base = self._get_window_data(data[i:(i+lb+lf)])
            X.append(tmp_X)
            y.append(tmp_y)
            base.append(tmp_base)
        return np.array(X), np.array(y), np.reshape(base, (-1,1))
    
    def _get_window_data(self, data):
        '''
        Normalize window to make fisrt value 1 and rest the ratio to the first value.
        '''
        lb, lf = self.lb, self.lf
        if self.normalize_window == 'ratio':
            X, y, base = data[:lb]/data[0], data[-lf:]/data[0], data[0]
        elif self.normalize_window == 'percent_change':
            X, y, base = (data[:lb]/data[0]-1)*100, (data[-lf:]/data[0]-1)*100, data[0]
        elif self.normalize_window is None:
            X, y, base = data[:lb], data[-lf:], 1
        else:
            raise ValueError('Unrecognizable normalization method.')
        return np.reshape(X, (-1,1)), np.array(y), np.array(base)

    
    def _train_val_split(self, X, y, base):
        '''
        Split training and validation data.
        '''
        split = self.train_val_split
        X_train, X_val = X[:int(len(X)*split)], X[int(len(X)*split):]
        y_train, y_val = y[:int(len(y)*split)], y[int(len(y)*split):]
        base_train, base_val = base[:int(len(base)*split)], base[int(len(base)*split):]
        return (X_train, y_train, base_train), (X_val, y_val, base_val)

        