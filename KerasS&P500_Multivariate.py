import pandas as pd
import numpy as np
import re
from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tabulate import tabulate

graphs = [3]
EPOCHS = 2
MINIBATCH_SIZE = 64
VERBOSE = True
INTERVAL_IN_DAYS = 1


# load dataset
def parser(x):
    # "Aug 09, 2018"
    return pd.datetime.strptime(x, '%b %d, %Y')


# create a differenced series
def pct_difference(in_dataset, target_y_col_nm):
    print(tabulate(in_dataset.sample(3), in_dataset.columns.values))
    history = in_dataset[[target_y_col_nm]]
    out_dataset = in_dataset.copy().apply(np.log)
    print(tabulate(out_dataset.tail(), out_dataset.columns.values))
    out_dataset_columns = out_dataset.columns.values
    offset = int(len(out_dataset_columns)/2)
    for col in range(offset):
        col_names = [out_dataset_columns[col], out_dataset_columns[col+offset]]
        is_target = [False, False]
        for i, col_name in enumerate(col_names):
            if re.search("\(t\)", col_name):
                is_target[i] = True

    drop_columns = [col for col in all_columns if re.search("\(t\)", col)]
    drop_columns.remove(target_y_col_nm)

    out_dataset[target_y_col_nm] = np.log(out_dataset[target_y_col_nm]) / np.log(out_dataset[offset_column_name])
    out_dataset.drop(columns=drop_columns, inplace=True)
    if VERBOSE:
        print("Sample Differenced Data\n====================")
        print(tabulate(out_dataset.sample(5), out_dataset.columns.values))
    exit()
    return out_dataset, history


# invert differenced value
def inverse_difference(in_history, in_yhat, interval):
    print(in_yhat)
    print('In History Shape: ', in_history.shape)
    in_yhat = pd.DataFrame(in_yhat)
    prediction_count = in_yhat.shape[0]
    in_history = in_history.shift(interval)
    print('Shifted History Shape: ', in_history.shape)
    in_history = in_history.iloc[-prediction_count:, :]
    print('Truncated History Shape: ', in_history.shape)
    print(in_yhat)
    print(in_history)
    results = in_history * in_yhat
    print(results)

    print(results.iloc[:4, :])
    print(results.iloc[-4:, :])
    return results


# convert series to supervised learning
def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(i))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-i))
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

supervised_data = series_to_supervised(mv_time_series, n_in=1, n_out=INTERVAL_IN_DAYS, dropnan=True)
if VERBOSE:
    print(tabulate(supervised_data.head(), supervised_data.columns.values))

pct_change_data, historical_data = pct_difference(supervised_data, r'var4(t)')

print(tabulate(mv_time_series.tail(), mv_time_series.columns.values))
print(mv_time_series.shape[0])
print(tabulate(historical_data.tail(), historical_data.columns.values))
print(historical_data.shape[0])
exit()
# Don't need the encoder because none of the fields are categorical
# scaler = MinMaxScaler(feature_range=(0, 1))
# if VERBOSE: print("supervised_data.shape: %s \nThis is what is sent to the scaler." % str(pct_change_data.shape))
# scaled = scaler.fit_transform(pct_change_data)
# if VERBOSE:
#     print(tabulate(scaled[0:4, :], pct_change_data.columns.values))
#     print("scaled.shape: ", scaled.shape)

num_values = pct_change_data.shape[0]
split_line = int(num_values*0.25)

train = pct_change_data.values[:split_line, :]
test = pct_change_data.values[split_line:, :]

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
print(test_X[0: 4, :])
print(test_y[0: 4])
print(historical_data[0:4])

inv_yhat = inverse_difference(historical_data, yhat, INTERVAL_IN_DAYS)

# invert scaling for actual
# inv_yhat = inv_yhat.reshape((len(test_y), 1))
# inv_y = np.concatenate((test_X, test_y), axis=1)
# inv_y = scaler.inverse_transform(inv_y)
# inv_y = inv_y[:, -1]
# print(test_X)
# calculate RMSE
rmse = np.sqrt(mean_squared_error(historical_data, inv_yhat))
print('Test RMSE: %.3f' % rmse)
exit()
length_of_test = test_y.shape[0]*-1
previous_price = mv_time_series['Price'][length_of_test:]
print(previous_price)
output = pd.DataFrame(list(zip(mv_time_series.index.values[length_of_test:],
                               previous_price,
                               previous_price*test_y,
                               previous_price*inv_yhat,
                               (previous_price*inv_yhat-inv_y*previous_price)*100/(previous_price*inv_y))),
                      columns=['Date',
                               'Previous Price',
                               'Actual',
                               'Predicted',
                               'Difference %']).set_index('Date')
print(tabulate(output.tail(5), output.columns.values))
print(mv_time_series.tail(1))
