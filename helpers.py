from statsmodels.tsa.seasonal import seasonal_decompose
from pmdarima import auto_arima
import pandas as pd

def strToList(s):
    s = s.split(',')
    l = [int(element) for element in s]
    return l

# def getCol(df):
#     return list(df.columns)

# def checkColName(l, name):
#     return name in l

# def colToList(df, name):
#     return df[name].tolist()

def listToDf(l, freq):
    dates = pd.date_range('1800-1-1', periods = len(l), freq= freq)
    df = pd.DataFrame({'dates': dates, 'data': l})
    df = df.set_index('dates')
    return df

def decompose(df, model):
    result = seasonal_decompose(df, model=model)
    return result

def gridSearch(data, seasonal, m):
    model = auto_arima(data, m= m , seasonal= seasonal)
    return model

def trainTestSplit(data, train_num):
    total_len = len(data.index)
    train = data.iloc[0:train_num]
    test = data.iloc[train_num:total_len]
    return train, test

def fitModel(model, train):
    model.fit(train)
    return model

def forecast(model, period):
    forecast = model.predict(n_periods=period)
    return forecast