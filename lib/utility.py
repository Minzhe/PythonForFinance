###########################################################################
###                              utility.py                             ###
###########################################################################

import matplotlib.pyplot as plt
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
