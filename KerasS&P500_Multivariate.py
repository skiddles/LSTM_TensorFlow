import pandas as pd
import numpy as np
import math
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tabulate import tabulate

graphs = []
EPOCHS = 2
MINIBATCH_SIZE = 16
VERBOSE = False


# load dataset
def parser(x):
    # "Aug 09, 2018"
    return pd.datetime.strptime(x, '%b %d, %Y')


# create a differenced series
def pct_difference(in_dataset, interval=1):
    in_dataset = in_dataset.apply(np.log)
    out_dataset = in_dataset.copy()
    out_dataset_columns = out_dataset.columns.values
    in_dataset = in_dataset.shift(periods=interval, axis=0)
    out_dataset = pd.merge(out_dataset, in_dataset, left_index=True, right_index=True, suffixes=('', '_lagged'))
    for column in out_dataset_columns:
        lagged_column = '_'.join([column, 'lagged'])
        out_dataset[column] = out_dataset[column] / out_dataset[lagged_column]
    out_dataset = out_dataset[out_dataset_columns]
    return out_dataset


# invert differenced value
def inverse_difference(history, yhat, interval=1):
    return history[-interval] * yhat


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    return agg


mv_time_series = pd.read_csv('./data/S&P 500 Historical Data.csv',
                             header=0,
                             parse_dates=[0],
                             index_col=0,
                             squeeze=True,
                             thousands=',',
                             usecols=[0, 1, 2, 3, 4],
                             date_parser=parser)

mv_time_series.sort_index(inplace=True)
column_names = list(mv_time_series.columns.values)
target = "Price"
column_names.pop(0)
column_names.append(target)

mv_time_series = mv_time_series[column_names]

values = mv_time_series.values

groups = [0, 1, 2, 3]
if 1 in graphs:
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(values[:, group])
        pyplot.title(mv_time_series.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

# This section was working
pct_change_data = pct_difference(mv_time_series, interval=1)

supervised_data = series_to_supervised(pct_change_data, n_in=1, n_out=1, dropnan=True)
supervised_data.drop(supervised_data.columns[[4, 5, 6]], axis=1, inplace=True)
if VERBOSE: print(tabulate(supervised_data.head(), supervised_data.columns.values))

# Don't need the encoder because none of the fields are categorical
scaler = MinMaxScaler(feature_range=(0, 1))
if VERBOSE: print("supervised_data.shape: %s \nThis is what is sent to the scaler." % str(supervised_data.shape))
scaled = scaler.fit_transform(supervised_data)
if VERBOSE:
    print(tabulate(scaled[0:4, :], supervised_data.columns.values))
    print("scaled.shape: ", scaled.shape)

num_values = supervised_data.shape[0]
split_line = int(num_values*0.25)

train = supervised_data.values[:split_line, :]
test = supervised_data.values[split_line:, :]

# split into input and outputs
train_X, train_y = train[:, :-1], train[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]
if VERBOSE:
    print("train_X")
    print(train_X[0:5, :])
    print("train_y")
    print(train_y[0:5])


# reshape input to be 3D [samples, timesteps, features]
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))

if VERBOSE:
    print("train_X.shape:, ", train_X.shape)
    print("train_y.shape: ", train_y.shape)


# design network
model = Sequential()
model.add(LSTM(50, input_shape=(train_X.shape[1], train_X.shape[2])))
model.add(Dense(1))
model.compile(loss='mae', optimizer='adam')
# fit network
history = model.fit(train_X, train_y,
                    epochs=EPOCHS,
                    batch_size=MINIBATCH_SIZE,
                    validation_data=(test_X, test_y),
                    verbose=2,
                    shuffle=False)

# plot history
if 2 in graphs:
    pyplot.plot(history.history['loss'], label='train')
    pyplot.plot(history.history['val_loss'], label='test')
    pyplot.legend()
    pyplot.show()

# make a prediction
if VERBOSE: print("test_X.shape: %s sent to predict" % str(test_X.shape))
yhat = model.predict(test_X)
if VERBOSE: print("test_X.shape[2]: ", test_X.shape[2])
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))
print(test_X.shape)
# invert scaling for forecast
print("yhat.shape: ", yhat.shape)
print(test_X[:, :].shape)

# inv_yhat = np.concatenate((test_X[:, :-1], yhat), axis=1)
inv_yhat = np.concatenate((test_X, yhat), axis=1)
print("inv_yhat.shape: ", inv_yhat.shape)
inv_yhat = scaler.inverse_transform(inv_yhat)
inv_yhat = inv_yhat[:, 0]
# invert scaling for actual
test_y = test_y.reshape((len(test_y), 1))
inv_y = np.concatenate((test_X, test_y), axis=1)
inv_y = scaler.inverse_transform(inv_y)
inv_y = inv_y[:, -1]
print(test_X)
# calculate RMSE
rmse = np.sqrt(mean_squared_error(inv_y, inv_yhat))
print('Test RMSE: %.3f' % rmse)
length_of_test = inv_y.shape[0]*-1
previous_price = mv_time_series['Price'][length_of_test:]
print(previous_price)
output = pd.DataFrame(list(zip(mv_time_series.index.values[length_of_test:],
                               previous_price,
                               previous_price*inv_y,
                               previous_price*inv_yhat,
                               (previous_price*inv_yhat-inv_y*previous_price)*100/(previous_price*inv_y))),
                      columns=['Date',
                               'Previous Price',
                               'Actual',
                               'Predicted',
                               'Difference %']).set_index('Date')
print(tabulate(output.tail(5), output.columns.values))
print(mv_time_series.tail(1))
