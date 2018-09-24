import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tabulate import tabulate

graphs = [5]
EPOCHS = 200
MINIBATCH_SIZE = 24
VERBOSE = True
INTERVAL_IN_DAYS = 1
TARGET_COL = 'SP500_Close'


# load dataset
def parser(x):
    # "Aug 09, 2018"
    # format = '%b %d, %Y'

    # 1/1/2000
    date_format = '%m/%d/%Y'
    return pd.datetime.strptime(x, date_format)


# create a differenced series
def pct_difference(in_dataset, target_y_col_nm, return_target):
    return_target_col_name = ''
    history = in_dataset[[target_y_col_nm]]
    out_dataset = in_dataset.copy().apply(np.log)
    out_dataset_columns = out_dataset.columns.values
    offset = int(len(out_dataset_columns)/2)
    for col in range(offset):
        col_names = [out_dataset_columns[col], out_dataset_columns[col+offset]]
        target_col = None
        interval_col = None
        new_col_name = None
        for i, col_name in enumerate(col_names):
            m = re.search("(var[0-9]+\()(t)(\))", col_name)
            if m is not None:
                target_col = col_name
                new_col_name = 'delta'.join([m.group(1), m.group(3)])
                if target_col == target_y_col_nm:
                    return_target_col_name = new_col_name
            else:
                interval_col = col_name
        out_dataset[new_col_name] = out_dataset[target_col] / out_dataset[interval_col]
    drop_columns = in_dataset.columns.values
    if return_target:
        drop_columns.remove(target_y_col_nm)
    else:
        return_target_col_name = target_y_col_nm

    out_dataset.drop(columns=drop_columns, inplace=True)
    if VERBOSE:
        print("Sample Differenced Data\n====================")
        print(tabulate(out_dataset.tail(5), out_dataset.columns.values))
    return out_dataset, return_target_col_name, history


# invert differenced value
def inverse_difference(in_history, in_yhat, interval):
    yhat = pd.DataFrame(in_yhat,
                        columns=in_history.columns.values,
                        index=in_history.tail(in_yhat.shape[0]).index.values)

    in_history = in_history.tail(in_yhat.shape[0]+interval).copy()
    prediction_count = yhat.shape[0]
    in_history = in_history.shift(interval)
    in_history = in_history.iloc[-prediction_count:, :]
    results = in_history * yhat
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


mv_time_series = pd.read_csv('./data/SP500_VIX_20000101_20180919_3.csv',
                             header=0,
                             parse_dates=[0],
                             index_col=0,
                             squeeze=True,
                             thousands=',',
                             usecols=[0, 1, 2, 3, 4, 6, 10],
                             date_parser=parser)

mv_time_series.sort_index(inplace=True)
column_names = list(mv_time_series.columns.values)
column_names.remove(TARGET_COL)
column_names.append(TARGET_COL)

mv_time_series = mv_time_series[column_names]
values = mv_time_series.values

groups = [3, 4, 5]
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

supervised_data = series_to_supervised(mv_time_series,
                                       n_in=1,
                                       n_out=INTERVAL_IN_DAYS,
                                       dropnan=True)
if VERBOSE:
    print(tabulate(supervised_data.head(), supervised_data.columns.values))

dif_tgt_col_nm = "var%i(t)" % mv_time_series.shape[1]

pct_change_data, target_col_nm, historical_data = pct_difference(supervised_data, dif_tgt_col_nm, return_target=False)

supervised_data2 = series_to_supervised(pct_change_data, n_in=1, n_out=INTERVAL_IN_DAYS, dropnan=True)
for col in supervised_data2.columns.values:
    target_day_col_name = re.search('var[0-9]+\(t\)', col)
    if col != target_col_nm and target_day_col_name:
        supervised_data2.drop(columns=col, inplace=True)

print(tabulate(supervised_data2.tail(), supervised_data2.columns.values))

groups = [3, 4, 5]
if 3 in graphs:
    i = 1
    # plot each column
    pyplot.figure()
    for group in groups:
        pyplot.subplot(len(groups), 1, i)
        pyplot.plot(supervised_data2.values[:, group])
        pyplot.title(supervised_data2.columns[group], y=0.5, loc='right')
        i += 1
    pyplot.show()

print(supervised_data2.tail())

time_series_data = supervised_data2

num_values = time_series_data.shape[0]

split_line = int(num_values*0.25)

train = time_series_data.values[:split_line, :]

test = time_series_data.values[split_line:, :]

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
if VERBOSE:
    print("test_X.shape: %s sent to predict" % str(test_X.shape))

yhat = model.predict(test_X)

if VERBOSE:
    print("test_X.shape[2]: ", test_X.shape[2])
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))

inv_yhat = inverse_difference(historical_data, yhat, INTERVAL_IN_DAYS)

# final_data = pd.merge(historical_data.tail(inv_yhat.shape[0]),
#                       inv_yhat,
#                       left_index=True,
#                       right_index=True,
#                       suffixes=('_actual', '_predicted'))

final_data = pd.merge(mv_time_series, inv_yhat, left_index=True, right_index=True)

final_data['Predicted Close'] = final_data[dif_tgt_col_nm]
final_data.drop(columns = dif_tgt_col_nm, inplace=True)
final_data['Pct Difference'] = (final_data['Predicted Close'] - final_data[TARGET_COL])*100 / final_data[TARGET_COL]
rmse = np.sqrt(mean_squared_error(historical_data.tail(inv_yhat.shape[0]), inv_yhat))
print('Test RMSE: %.3f' % rmse)

# length_of_test = test_y.shape[0]*-1
# previous_price = mv_time_series['Price'][length_of_test:]
print(tabulate(final_data.tail(15), final_data.columns.values))

print("Min Diff: %0.3f%s, Mean Diff: %0.3f%s, StdDev Diff: %0.3f%s, Max Diff: %0.3f%s" %
      (final_data['Pct Difference'].abs().min(), "%",
       final_data['Pct Difference'].mean(), "%",
       final_data['Pct Difference'].std(), "%",
       final_data['Pct Difference'].abs().max(), "%"))

if 4 in graphs:
    pyplot.plot(final_data[TARGET_COL], label='Actual')
    pyplot.plot(final_data['Predicted Close'], label='Predicted')
    pyplot.legend()
    pyplot.show()

if 5 in graphs:
    pyplot.plot(final_data['Pct Difference'], label='Percent Difference')
    pyplot.legend()
    pyplot.show()
