import pandas as pd
import yfinance as yf
import pandas_datareader as web
import matplotlib
import datetime as dt

test = yf.Ticker("BTC-USD")
test = test.history(period="max")

# del test["Dividends"]
# del test["Stock Splits"]

test.drop(['Dividends','Stock Splits'], axis=1, inplace=True)
test["Tomorrow"] = test["Close"].shift(-1)
print(test.head())


