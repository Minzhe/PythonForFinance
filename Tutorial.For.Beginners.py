###########################################################################
###               Python for Finance Tutorial for Beginners             ###
###########################################################################
# https://github.com/datacamp/datacamp-community-tutorials/blob/master/Python%20Finance%20Tutorial%20For%20Beginners/Python%20For%20Finance%20Beginners%20Tutorial.ipynb

import pandas as pd
import numpy as np
import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from pandas_datareader import data as pdr
from pandas.core import datetools
from pandas import tseries
import statsmodels.api as sm
import fix_yahoo_finance as yf
import quandl
sns.set()

################   function  #################
def getStocks(tickers, startdate, enddatate):
    '''
    Get stocks price.
    '''
    def getStock(ticker, startdate, enddatate):
        return pdr.get_data_yahoo(ticker, start=startdate, end=enddatate)
    stocks = map(lambda x: getStock(x, startdate, enddatate), tickers)
    return pd.concat(stocks, keys=tickers, names=['Ticker', 'Date'])

#########################   main  ##############################
# >>>>>>>>>>>>>>>>>>>>> Time series data <<<<<<<<<<<<<<<<<<<<<<< #
aapl = pdr.get_data_yahoo('AAPL', start=datetime.datetime(1998, 10, 1), end=datetime.datetime.today().date())
aapl = quandl.get('WIKI/AAPL', start_date=datetime.datetime(1998, 10, 1), end_date=datetime.datetime.today().date())
# aapl.head()
# aapl.loc['2007']
# aapl.sample(20)
# aapl.Open - aapl.Close
aapl['Close'].plot(grid=True)
plt.plot(aapl['Close'], 'b')
plt.plot(aapl['Adj Close'], 'g')

# >>>>>>>>>>>>>>>>>>>>> Financial Analysis <<<<<<<<<<<<<<<<<<<<<<< #
# daily returns
daily_returns = aapl['Adj Close'].pct_change()
daily_returns = aapl['Adj Close'] / aapl['Adj Close'].shift(1) - 1
daily_returns.fillna(0, inplace=True)
plt.plot(daily_returns)
sns.distplot(daily_returns, bins=50, hist=True, kde=False)

# bussiness month
monthly_returns = aapl['Adj Close'].resample('BM').apply(lambda x: x[-1]).pct_change()
plt.plot(monthly_returns)

# quarter
quarterly_returns = aapl['Adj Close'].resample('4M').mean().pct_change()
plt.plot(quarterly_returns)

# cumulative daily returns
cum_daily_return = (1 + daily_returns).cumprod()
plt.plot(cum_daily_return)

# cumulative month returns
cum_monthly_returns = cum_daily_return.resample('M').mean()
plt.plot(cum_monthly_returns)

# >>>>>>>>>>>>>>>>>>>>> Compare stocks <<<<<<<<<<<<<<<<<<<<<<< #
tickers = ['AAPL', 'MSFT', 'IBM', 'GOOG', 'AMZN']
all_data = getStocks(tickers, datetime.datetime(1998, 10, 1), datetime.datetime.today().date())

# daily return
daily_close_px = all_data['Adj Close'].reset_index().pivot('Date', 'Ticker', 'Adj Close')
daily_returns = daily_close_px.pct_change()
daily_returns.fillna(0, inplace=True)
plt.plot(daily_close_px)
daily_returns.hist(bins=50, sharex=True, figsize=(12,8))

# scatter plot
pd.plotting.scatter_matrix(daily_returns, diagonal='kde', alpha=0.1, figsize=(12,12))

# >>>>>>>>>>>>>>>>>>>>> Moving average <<<<<<<<<<<<<<<<<<<<<<< #
aapl['ma 42'] = aapl['Adj Close'].rolling(window=40).mean()
aapl['ma 252'] = aapl['Adj Close'].rolling(window=250).mean()
plt.plot(aapl[['Adj Close', 'ma 42', 'ma 252']])

# >>>>>>>>>>>>>>>>>>>>> Volatility <<<<<<<<<<<<<<<<<<<<<<< #
period = 75
vol = daily_returns.rolling(period).std() * np.sqrt(period)
plt.plot(vol)

# >>>>>>>>>>>>>>>>>>>>> Ordinary Least-Squares Regression (OLS) <<<<<<<<<<<<<<<<<<<<<<< #
aapl_returns = daily_returns['AAPL']
msft_returns = daily_returns['MSFT']

X = sm.add_constant(aapl_returns)
model = sm.OLS(msft_returns, X).fit()
model.summary()

plt.plot(aapl_returns, msft_returns, 'r.')
ax = plt.axis()
x = np.linspace(ax[0], ax[1] + 0.01)
plt.plot(x, model.params[0] + model.params[1] * x, 'b', lw=2)
plt.axis('tight')
plt.xlabel('Apple Returns')
plt.ylabel('Microsoft returns')

msft_returns.rolling(window=252).corr(aapl_returns).plot()

# >>>>>>>>>>>>>>>>>>>>> Trading Strategy <<<<<<<<<<<<<<<<<<<<<<< #
short_window = 40
long_window = 100

signals = pd.DataFrame(index=aapl.index)
signals['signal'] = 0.0
signals['short_ma'] = aapl['Close'].rolling(window=short_window, min_periods=1, center=False).mean()
signals['long_ma'] = aapl['Close'].rolling(window=long_window, min_periods=1, center=False).mean()
signals['signal'][short_window:] = np.where(signals['short_ma'][short_window:] > signals['long_ma'][short_window:], 1.0, 0.0)
signals['positions'] = signals['signal'].diff()

plt.plot(aapl['Close'], color='k', lw=2.)
plt.plot(signals[['short_ma', 'long_ma']], lw=2.)

# plot buy signal
plt.plot(signals.loc[signals.positions == 1.0].index, 
         signals.short_ma[signals.positions == 1.0],
         '^', markersize=4, color='g')

# plot sell signals
plt.plot(signals.loc[signals.positions == -1.0].index, 
         signals.short_ma[signals.positions == -1.0],
         'v', markersize=4, color='r')

# >>>>>>>>>>>>>>>>>>>>> Backtesting A Strategy <<<<<<<<<<<<<<<<<<<<<<< #
initial_capital= float(100000.0)
positions = pd.DataFrame(index=signals.index).fillna(0.0)
positions['AAPL'] = 100*signals['signal']
portfolio = positions.multiply(aapl['Adj Close'], axis=0)
pos_diff = positions.diff()
portfolio['holdings'] = (positions.multiply(aapl['Adj Close'], axis=0)).sum(axis=1)
portfolio['cash'] = initial_capital - (pos_diff.multiply(aapl['Adj Close'], axis=0)).sum(axis=1).cumsum()
portfolio['total'] = portfolio['cash'] + portfolio['holdings']

plt.plot(portfolio['total'], color='b', lw=2.)
plt.plot(signals.loc[signals.positions == 1.0].index, 
         portfolio.total[signals.positions == 1.0],
         '^', markersize=4, color='g')
plt.plot(signals.loc[signals.positions == -1.0].index, 
         portfolio.total[signals.positions == -1.0],
         'v', markersize=4, color='r')

# >>>>>>>>>>>>>>>>>>>>> Evaluating Moving Average Crossover Strategy <<<<<<<<<<<<<<<<<<<<<<< #
# Sharpe ratio
returns = portfolio['returns']
sharpe_ratio = np.sqrt(252) * (returns.mean() / returns.std())

# Maximum Drawdown
window = 252
rolling_max = aapl['Adj Close'].rolling(window, min_periods=1).max()
daily_drawdown = aapl['Adj Close']/rolling_max - 1.0
max_daily_drawdown = daily_drawdown.rolling(window, min_periods=1).min()

plt.plot(daily_drawdown)
plt.plot(max_daily_drawdown)
