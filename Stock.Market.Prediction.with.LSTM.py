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
from lib.LSTM_model import LSTM_model
from lib import utility as util
from lib.dataGenerator import dataGenerator
sns.set()

#############################    main    ###################################
data = pd.read_csv('data/all_stocks_5yr.csv', index_col=0, parse_dates=True)
aapl = data.loc[data['Name'] == 'AAPL', 'close']
jpm = data.loc[data['Name'] == 'JPM', 'close']
# f, (ax1, ax2) = plt.subplots(2)
# util.plot_series(truth=aapl, title='AAPL', ax=ax1)
# util.plot_series(truth=jpm, title='JPM', ax=ax2)
# plt.show()

# >>>>>>>>>>>>  Split data and normalize  <<<<<<<<<<<<<< #
# apple
aapl_dgr = dataGenerator(data=aapl, lb=14, normalize_window='percent_change', train_val_split=0.8, test_windows=30)
X1_train, y1_train, base1_train = aapl_dgr.get_train_data()
X1_val, y1_val, base1_val = aapl_dgr.get_val_data()
X1_test, y1_test, base1_test = aapl_dgr.get_test_data()
# jpmorgan
jpm_dgr = dataGenerator(data=jpm, lb=14, normalize_window='percent_change', train_val_split=0.8, test_windows=30)
X2_train, y2_train, base2_train = jpm_dgr.get_train_data()
X2_val, y2_val, base2_val = jpm_dgr.get_val_data()
X2_test, y2_test, base2_test = jpm_dgr.get_test_data()

# >>>>>>>>>>>>  Build LSTM model  <<<<<<<<<<<<<< #
# train model on apple
lstm = LSTM_model(input_shape=(14,1), structure='1-layer-lstm')
trace = lstm.train(X1_train, y1_train, save_path='lib/model/lstm.h5', validation_data=(X1_val, y1_val), epochs=200, patience=20, shuffle=False)
lstm.loadModel('lib/model/lstm.h5')
f, ax = util.plot_trace(trace)
plt.show()

# >>>>>>>>>>>>  Prediction  <<<<<<<<<<<<<< #
# apple
pred1_train = aapl_dgr.inverse_transform(lstm.predict_one_step(X1_train), base1_train)
pred1_val = aapl_dgr.inverse_transform(lstm.predict_one_step(X1_val), base1_val)
pred1_test = aapl_dgr.inverse_transform(lstm.predict_multi_steps(X1_test, steps=30), base1_test)
# jpmorgan
pred2_train = jpm_dgr.inverse_transform(lstm.predict_one_step(X2_train), base2_train)
pred2_val = jpm_dgr.inverse_transform(lstm.predict_one_step(X2_val), base2_val)
pred2_test = jpm_dgr.inverse_transform(lstm.predict_multi_steps(X2_test, steps=30), base2_test)
# plot
f, (ax1, ax2) = plt.subplots(2)
util.plot_series(truth=aapl[14:], title='AAPL', pred=(pred1_train, pred1_val, pred1_test), ax=ax1)
util.plot_series(truth=jpm[14:], title='JPM', pred=(pred2_train, pred2_val, pred2_test), ax=ax2)
plt.show()