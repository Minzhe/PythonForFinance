###########################################################################
###                    Stock Market Prediction with LSTM                ###
###########################################################################
# https://github.com/thushv89/datacamp_tutorials/blob/master/Reviewed/lstm_stock_market_prediction.ipynb

from pandas_datareader import data as pdr
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import datetime as dt
import numpy as np
from subprocess import check_output
from lib.LSTM_model import LSTM_model
from lib.utility import MinMaxScaler
from lib.dataGenerator import dataGenerator

sns.set()
check_output(['ls', 'data']).decode('utf-8')

#############################    main    ###################################
data = pd.read_csv('data/all_stocks_5yr.csv', index_col=0, parse_dates=True)
mmm = data.loc[data['Name'] == 'MMM', 'close']
# fig, ax = plot(mmm)
# plt.show()

# >>>>>>>>>>>>  Split data and normalize  <<<<<<<<<<<<<< #
scaler = MinMaxScaler(mmm)
mmm = scaler.transform(mmm)
data_mmm = dataGenerator(data=mmm, lb=7, train_val_split=0.8, test_windows=30)
X_train, y_train = data_mmm.get_train_data()
X_val, y_val = data_mmm.get_val_data()
X_test, y_test = data_mmm.get_test_data()

# >>>>>>>>>>>>  Build LSTM model  <<<<<<<<<<<<<< #
lstm = LSTM_model(input_shape=(7,1), structure='1-layer-lstm')
trace = lstm.train(X_train, y_train, validation_data=(X_val, y_val), patience=5, shuffle=False)
plt.plot(trace.history['loss'])
plt.plot(trace.history['val_loss'])
plt.show()

# >>>>>>>>>>>>  Prediction  <<<<<<<<<<<<<< #
pred_train = lstm.predict_one_step(X_train)
pred_val = lstm.predict_one_step(X_val)
pred_test = lstm.predict_multi_steps(X_test, steps=30)
plt.plot(mmm[7:])
plt.plot(pd.Series(np.squeeze(np.concatenate([pred_train, pred_val, pred_test])), index=mmm[7:].index))
plt.show()