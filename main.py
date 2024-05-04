import pandas as pd
import yfinance as yf
import pandas_datareader as web
import matplotlib
import datetime as dt

msft = yf.Ticker("BTC-USD")
msft = msft.history(period="max")

print(msft.head())
print(msft.tail())

