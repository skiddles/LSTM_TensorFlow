from pandas import read_csv
from pandas import datetime
from pandas import DataFrame, Series
from pandas import concat
from sklearn.preprocessing import MinMaxScaler
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from sklearn.metrics import mean_squared_error
from math import sqrt
from matplotlib import pyplot
import numpy
from colorama import Fore, Style


# load dataset
def parser(x):
    return datetime.strptime('190'+x, '%Y-%m')


# create a differenced series
def difference(dataset, interval=1):
    diff = list()
    for i in range(interval, len(dataset)):
        value = dataset[i] - dataset[i - interval]
        diff.append(value)
    return Series(diff)


# frame a sequence as a supervised learning problem
def timeseries_to_supervised(data, lag=1):
    df = DataFrame(data)
    columns = [df.shift(i) for i in range(1, lag+1)]
    columns.append(df)
    df = concat(columns, axis=1)
    df.fillna(0, inplace=True)
    return df


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return yhat + history[-interval]


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

# inverse scaling for a forecasted value
def invert_scale(scaler, X, value):
    new_row = [x for x in X] + [value]
    array = numpy.array(new_row)
    array = array.reshape(1, len(array))
    inverted = scaler.inverse_transform(array)
    return inverted[0, -1]


# scale train and test data to [-1, 1]
def scale(in_train, in_test):
    # fit scaler
    core_scaler = MinMaxScaler(feature_range=(-1, 1))
    core_scaler = core_scaler.fit(in_train)
    # transform train
    in_train = in_train.reshape(train.shape[0], train.shape[1])
    out_train_scaled = core_scaler.transform(in_train)
    # transform test
    in_test = in_test.reshape(in_test.shape[0], in_test.shape[1])
    out_test_scaled = core_scaler.transform(in_test)
    return core_scaler, out_train_scaled, out_test_scaled


series = read_csv('../Data/Shampoo/shampoo.csv',
                  header=0,
                  parse_dates=[0],
                  index_col=0,
                  squeeze=True,
                  date_parser=parser)


# transform data to be stationary
raw_values = series.values
diff_values = difference(raw_values, 1)

# transform data to be supervised learning
supervised = timeseries_to_supervised(diff_values, 1)
supervised_values = supervised.values

print(supervised_values.shape)
# split data into train and test-sets
train, test = supervised_values[0:-12], supervised_values[-12:]

# transform the scale of the data
scaler, train_scaled, test_scaled = scale(train, test)

print(train_scaled[0:4])

# fit the model
lstm_model = fit_lstm(train_scaled, 1, 3000, 4)

# forecast the entire training dataset to build up state for forecasting
train_reshaped = train_scaled[:, 0].reshape(len(train_scaled), 1, 1)
lstm_model.predict(train_reshaped, batch_size=1)

# walk-forward validation on the test data
predictions = list()
for i in range(len(test_scaled)):
    # make one-step forecast
    X, y = test_scaled[i, 0:-1], test_scaled[i, -1]
    yhat = forecast_lstm(lstm_model, 1, X)
    # invert scaling
    yhat = invert_scale(scaler, X, yhat)
    # invert differencing
    yhat = inverse_difference(raw_values, yhat, len(test_scaled) + 1 - i)
    # store forecast
    predictions.append(yhat)
    expected = raw_values[len(train) + i + 1]
    print('Month=%d, Predicted=%f, Expected=%f' % (i + 1, yhat, expected))

# report performance
rmse = sqrt(mean_squared_error(raw_values[-12:], predictions))
print('Test RMSE: %.3f' % rmse)
# line plot of observed vs predicted
pyplot.plot(raw_values[-12:])
pyplot.plot(predictions)
pyplot.show()
