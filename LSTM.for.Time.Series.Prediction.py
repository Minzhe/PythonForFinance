###########################################################################
###                    LSTM for Time Series Prediction                  ###
###########################################################################

import numpy as np
import pandas as pd
from scipy import stats
import matplotlib.pyplot as plt
import seaborn as sns
from lib.LSTM_model import LSTM_model
from lib import utility as util
from lib.dataGenerator import dataGenerator
sns.set()

######################  main  ########################
# >>>>>>>>>>>>  generate data  <<<<<<<<<<<<<< #
x = np.linspace(0, np.pi*20, 1000)
data = pd.Series(np.sin(x), index=range(len(x)))
# f, ax = plt.subplots()
# util.plot_series(truth=data, ax=ax, title='sin')
# plt.show()

# >>>>>>>>>>>>  Split data  <<<<<<<<<<<<<< #
dgr = dataGenerator(data=data, lb=20, lf=1, normalize_window=None, train_val_split=0.75, test_windows=500)
X_train, y_train, base_train = dgr.get_train_data()
X_val, y_val, base_val = dgr.get_val_data()
X_test, y_test, base_test = dgr.get_test_data()

# >>>>>>>>>>>>  Build LSTM model  <<<<<<<<<<<<<< #
# train model on apple
lstm = LSTM_model(input_shape=(20,1), output_dim=1, structure='1-layer-lstm')
trace = lstm.train(X_train, y_train, save_path='lib/model/sin_lstm.h5', validation_data=(X_val, y_val), epochs=200, patience=5, shuffle=False)
lstm.loadModel('lib/model/sin_lstm.h5')
# f, ax = util.plot_trace(trace)
# plt.show()

# >>>>>>>>>>>>  Prediction  <<<<<<<<<<<<<< #
pred_train = dgr.inverse_transform(lstm.predict_one_step(X_train), base_train)
pred_val = dgr.inverse_transform(lstm.predict_one_step(X_val), base_val)
pred_test = dgr.inverse_transform(lstm.predict_multi_steps(X_test, steps=500, stride=1), base_test)
f, ax = plt.subplots()
util.plot_series(truth=data[20:], ax=ax, title='sin', pred=(pred_train, pred_val, pred_test))
plt.show()