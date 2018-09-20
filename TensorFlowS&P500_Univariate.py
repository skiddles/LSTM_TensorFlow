import pandas as pd
# from pandas import datetime
# from pandas import DataFrame, Series
# from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import numpy as np
from colorama import Fore, Style
from tabulate import tabulate

# load dataset
def parser(x):
    # "Aug 09, 2018"
    return pd.datetime.strptime(x, '%b %d, %Y')

# create a differenced series
def pct_difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = np.log(dataset[i]) / np.log(dataset[i - interval])
        diff.append(value)
    return pd.Series(diff)


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = pd.DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = pd.concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return history[-interval] * yhat


# fit an LSTM network to training data
def fit_lstm(in_train, batch_size, nb_epoch, neurons):
    X, y = in_train[:, 0:-1], in_train[:, -1]
    X = X.reshape(X.shape[0], 1, X.shape[1])
    model = Sequential()
    model.add(LSTM(neurons, 
                   batch_input_shape=(batch_size, 
                                      X.shape[1], 
                                      X.shape[2]), 
                   stateful=True))
    
    model.add(Dense(1))
    
    model.compile(loss='mean_squared_error', 
                  optimizer='adam')
    
    for ix in range(nb_epoch):
        if ix == 0:
            print("Beginning training!")
        if ix % 100 == 0 and ix > 0:
            if ix % 1000 == 0:
                print(f"{Fore.BLUE}Running epoch: %i{Style.RESET_ALL}" % ix)
            else:
                print("Running epoch: %i" % ix)
        model.fit(X, y,
                  epochs=1,
                  batch_size=batch_size,
                  verbose=0,
                  shuffle=False)
        model.reset_states()
    return model


# make a one-step forecast
def forecast_lstm(model, batch_size, X):
    X = X.reshape(1, 1, len(X))
    yhat = model.predict(X, batch_size=batch_size)
    return yhat[0, 0]


time_series = pd.read_csv('./data/S&P 500 Historical Data.csv',
                          header=0,
                          parse_dates=[0],
                          index_col=0,
                          squeeze=True,
                          date_parser=parser)

print(time_series.columns.values)
time_series['Price_Text'] = time_series['Price']
time_series['Price'] = time_series['Price'].str.replace(',', '').astype(np.float32)
time_series.sort_index(inplace=True)
print(tabulate(time_series.head(), time_series.columns.values))
# transform data to be stationary
raw_values = time_series['Price'].values
diff_values = pct_difference(raw_values)
print(diff_values)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

num_values = supervised_values.shape[0]
split_line = int(num_values*0.5)

# split data into train and test-sets
train_scaled, test_scaled = supervised_values[0:split_line-1], supervised_values[split_line:]

print(train_scaled[0:4])

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 20, 10)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train_scaled) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[split_line+1:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[split_line+1:])
pyplot.plot(predictions)
pyplot.show()
