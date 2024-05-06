import pandas as pd
import yfinance as yf
import pandas_datareader as web
import matplotlib.pyplot as plt
import datetime as dt
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score


pd.set_option('display.max_columns',None)
pd.set_option('display.max_rows',None)
pd.set_option('display.max_colwidth',None)
pd.set_option('display.width',200)

def predict(train,test,predictors,model):
    model.fit(train[predictors],train["Target"])
    preds = model.predict_proba(test[predictors])[:,1]
    preds[preds >=.6] = 1
    preds[preds < .6] = 0
    preds = pd.Series(preds,index=test.index,name="Predictions")
    combined = pd.concat([test["Target"],preds],axis=1)
    return combined

def backtest(data,model,predictors,start=1000,step=250):
    all_predictions = []
    for i in range(start,data.shape[0],step): 
        train = data.iloc[0:i].copy()
        test = data.iloc[i:(i+step)].copy()
        predictions = predict(train,test,predictors,model)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)

underlying = yf.Ticker("BTC-USD")
underlying = underlying.history(period="max")
# underlying=underlying.loc["1980-01-01":].copy()

underlying.drop(['Dividends','Stock Splits'], axis=1, inplace=True)
underlying["Tomorrow"] = underlying["Close"].shift(-1)
underlying["Target"] = (underlying["Tomorrow"] > underlying["Close"]).astype(int)

model = RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)

predictors = ["Close","Volume","Open","High","Low"]
horizons = [2,5,60,90,180,250,1000]
new_predictors = []

for horizon in horizons:
    rolling_averages = underlying.rolling(horizon).mean()
    ratio_column = f"Close_Ratio_{horizon}"
    underlying[ratio_column] = underlying["Close"] / rolling_averages["Close"]

    trend_column = f"Trend_{horizon}"
    underlying[trend_column] = underlying.shift(1).rolling(horizon).sum()["Target"]

    new_predictors+= [ratio_column,trend_column]

underlying = underlying.dropna()

predictions = backtest(underlying,model,new_predictors)
print(predictions["Predictions"].value_counts())
print(precision_score(predictions["Target"],predictions["Predictions"]))
print(predictions.head())
# print(underlying.head())


# predictions = backtest(underlying,model,predictors)
# print(predictions.head())

# print(predictions["Predictions"].value_counts())
# print(precision_score(predictions["Target"],predictions["Predictions"]))
# print(predictions["Target"].value_counts()/predictions.shape[0])



# underlying = yf.Ticker("BTC-USD")
# underlying = underlying.history(period="max")

# # del test["Dividends"]
# # del test["Stock Splits"]

# underlying.drop(['Dividends','Stock Splits'], axis=1, inplace=True)
# underlying["Tomorrow"] = underlying["Close"].shift(-1)
# underlying["Target"] = (underlying["Tomorrow"] > underlying["Close"]).astype(int)

# model = RandomForestClassifier(n_estimators=100,min_samples_split=100,random_state=1)
# train = underlying.iloc[:-100]
# test = underlying.iloc[-100:]

# predictors = ["Close","Volume","Open","High","Low"]
# model.fit(train[predictors],train["Target"])
# preds = model.predict(test[predictors])
# preds = pd.Series(preds,index=test.index)
# print(precision_score(test["Target"],preds))
# combined = pd.concat([test["Target"],preds],axis=1)
# plt.plot(combined)
# plt.show()


# print(underlying.head())


