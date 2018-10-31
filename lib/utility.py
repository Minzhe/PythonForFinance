###########################################################################
###                              utility.py                             ###
###########################################################################

import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd

# >>>>>>>>>>>>>>>>>>  plotting  <<<<<<<<<<<<<<<<<<<< #
def plot(data, title='', xlabel='', ylabel=''):
    fig, ax = plt.subplots()
    ax.plot(data)
    ax.set_title(title)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    return fig, ax

# >>>>>>>>>>>>>>>>>>  normalization  <<<<<<<<<<<<<<<<<<<< #
class MinMaxScaler(object):
    def __init__(self, data):
        self.__max = max(data)
        self.__min = min(data)
        self.__range = self.__max - self.__min

    def transform(self, data):
        return (data - self.__min) / self.__range
    
    def inverse_transform(self, data):
        return data * self.__range + self.__min

def cal_pct_change(data):
    return np.array(pd.Series(data).pct_change().fillna(0))

def reverse_pct_change(data, base):
    data = data + 1
    data[0] = base
    return np.array(data.cumprod(data))

# >>>>>>>>>>>>>>>>>>  plot  <<<<<<<<<<<<<<<<<<<< #
def plot_trace(trace):
    '''
    Plot trace during training.
    '''
    f, ax = plt.subplots()
    ax.plot(np.log(trace.history['loss']), label='log_loss')
    ax.plot(np.log(trace.history['val_loss']), label='log_val_loss')

    return f, ax


def plot_series(truth, ax, title='', pred=None):
    '''
    Plot truth values with prediction in train, val and test data.
    '''
    # plot truth
    ax.plot(truth, label='Truth')
    ax.set_title(title)

    # plot prediction
    if pred is not None:
        train, val, test = pred
        train = np.squeeze(train)
        val = np.squeeze(val)
        test = np.squeeze(test)

        if len(train) + len(val) + len(test) != len(truth):
            raise ValueError('Truth and prediction length do not match.')
        train = pd.Series(train, index=truth.index[:len(train)])
        val = pd.Series(val, index=truth.index[len(train):-len(test)])
        test = pd.Series(test, index=truth.index[-len(test):])
          
        ax.plot(train, label='Train')
        ax.plot(val, label='Val')
        ax.plot(test, label='Test')
        ax.legend()
