import pandas as pd
import numpy as np
import re
from sklearn.metrics import mean_squared_error
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import MinMaxScaler
from sklearn.preprocessing import FunctionTransformer
from matplotlib import pyplot
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from tabulate import tabulate


graphs = [5]
EPOCHS = 2000
MINIBATCH_SIZE = 40
VERBOSE = False
INTERVAL_IN_DAYS = 60
TARGET_COL = 'SP500_AdjClose'
TRAIN_PCT = 0.6


MODEL_COLUMNS = {'SP500_Open': {'FUNCTIONAL_NAME': 'Market_Open',
                                'TRANSFORM': 'pipeline_to_stationary'},

                 'SP500_High': {'FUNCTIONAL_NAME': 'Market_High',
                                'TRANSFORM': 'pipeline_to_stationary'},

                 'SP500_Low': {'FUNCTIONAL_NAME': 'Market_Low',
                               'TRANSFORM': 'pipeline_to_stationary'},

                 'SP500_AdjClose': {'FUNCTIONAL_NAME': 'Market_AdjClose',
                                    'TRANSFORM': 'pipeline_to_stationary'},

                 'SP500_Volume': {'FUNCTIONAL_NAME': 'Market_Volume_Magnitude',
                                  'TRANSFORM': 'pipeline_to_stationary'},

                 'VIX_Close': {'FUNCTIONAL_NAME': 'VIX_Close_Magnitude',
                               'TRANSFORM': 'pipeline_to_stationary'}
                 }


# load dataset
def parser(x):
    # "Aug 09, 2018"
    # format = '%b %d, %Y'

    # 1/1/2000
    date_format = '%m/%d/%Y'
    return pd.datetime.strptime(x, date_format)


# convert series to supervised learning
def series_to_supervised(data, **kwargs):
    print("Beginning series_to_supervised() now.")
    # Takes a single column DataFrame
    # Returns a two column DataFrame where the first is lagged from the second by the interval
    n_in = 1
    n_out = 1
    interval_in_days = 1
    dropnan = True
    return_shifted_only = False
    if isinstance(data, np.ndarray):
        data = pd.DataFrame(data)
    for k, v in kwargs.items():
        if k == 'interval_in_days_in':
            interval_in_days = v
        elif k == 'dropnan':
            dropnan = v
        elif k == 'shifted_data_only':
            return_shifted_only = v

    n_vars = 1 if type(data) is list else data.shape[1]
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(data.shift(interval_in_days))
        names += [('var%d(t-%d)' % (j + 1, i)) for j in range(n_vars)]
    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(data.shift(-interval_in_days))
        if i == 0:
            names += [('var%d(t)' % (j + 1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j + 1, i)) for j in range(n_vars)]
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names
    if return_shifted_only:
        drop_column = [ix for ix, x in enumerate(agg.columns.values) if re.match('var[0-9]+\(t\)', x)]
        agg.drop(columns=agg.columns.values[drop_column], inplace=True)
    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    if VERBOSE:
        print("Supervising the data\n====================")
        print(tabulate(agg.tail(), agg.columns.values))
    return agg


# create a differenced series
def pct_difference(X):
    print("Beginning pct_difference() now.")
    if isinstance(X, np.ndarray):
        X = pd.DataFrame(X, columns=['var1(t-1)', 'var1(t)'])
    # print(X.columns.values)
    in_cols_as_dataset = X

    # Takes a DataFrame of two columns (varX(t-1), varX(t))
    # Returns a DataFrame of one columns (varX(delta))
    inbound_column_names = in_cols_as_dataset.columns.values
    out_dataset = in_cols_as_dataset.copy().apply(np.log)
    target_col = None
    interval_col = None
    new_col_name = None
    for i, col_name in enumerate(inbound_column_names):
        m = re.search("(var[0-9]+\()(t)(\))", col_name)
        if m is not None:
            target_col = col_name
            new_col_name = 'delta'.join([m.group(1), m.group(3)])
        else:
            interval_col = col_name
    out_dataset[new_col_name] = out_dataset[target_col] / out_dataset[interval_col]
    drop_columns = in_cols_as_dataset.columns.values

    out_dataset.drop(columns=drop_columns, inplace=True)

    if VERBOSE:
        print("Sample Differenced Data\n====================")
        print(tabulate(out_dataset.tail(5), out_dataset.columns.values))
    return out_dataset


# invert differenced value
def inverse_difference(in_history, in_yhat, interval):
    results = in_history * in_yhat
    return results


timeseries_to_supervised_kwargs = {'interval_in_days_in': INTERVAL_IN_DAYS,
                                   'dropnan': True}

pct_change_to_supervised_kwargs = {'interval_in_days_in': INTERVAL_IN_DAYS,
                                   'dropnan': True}

pipeline_to_stationary = Pipeline([('timeseries_to_supervised', FunctionTransformer(
                                       series_to_supervised, kw_args=timeseries_to_supervised_kwargs)),
                                   ('supervised_to_pct_change', FunctionTransformer(pct_difference)),
                                   ('pct_change_to_supervised', FunctionTransformer(
                                       series_to_supervised, kw_args=pct_change_to_supervised_kwargs))])


magnitude_to_supervised_kwargs = {'interval_in_days_in': INTERVAL_IN_DAYS,
                                  'dropnan': False,
                                  'shifted_data_only': True}

scaler = MinMaxScaler(feature_range=(0, 1))
pipeline_to_min_max = Pipeline([('to_min_max', scaler),
                                ('shift_data',  FunctionTransformer(
                                    series_to_supervised, kw_args=magnitude_to_supervised_kwargs))])

mv_time_series = pd.read_csv('./data/SP500_VIX_20000101_20180919_3.csv',
                             header=0,
                             parse_dates=[0],
                             index_col=0,
                             squeeze=True,
                             thousands=',',
                             date_parser=parser)

mv_time_series.sort_index(inplace=True)
mv_time_series.reset_index(inplace=True)

working_data_frame = mv_time_series.copy()
output_data_frame = pd.DataFrame(index=mv_time_series.index.values)
output_data_frame['Date'] = mv_time_series['Date']

# First, filter the columns to used columns and apply the transform process based on the column definitions
column_names = list(MODEL_COLUMNS.keys())
working_data_frame = working_data_frame[column_names]


for col in column_names:
    # Get the transformations
    prep_dict = MODEL_COLUMNS[col]
    transform = prep_dict['TRANSFORM']
    working_col_nm = prep_dict['FUNCTIONAL_NAME']
    print("Sending %s to pipeline" % col)
    if transform == 'pipeline_to_stationary':
        output = pipeline_to_stationary.fit_transform(working_data_frame[[col]])
    elif transform == 'pipeline_to_min_max':
        output = pipeline_to_min_max.fit_transform(working_data_frame[[col]])
    output_data_frame = pd.merge(output_data_frame, output, left_index=True, right_index=True)
    for out_col in output_data_frame.columns.values:
        col_match = re.search(r'var[0-9]+(\([^)]+?\))', out_col)
        if col_match is not None:
            is_target = False
            if col == TARGET_COL:
                is_target = True
            old_col_name = col_match.group(0)
            old_col_suffix = col_match.group(1)
            new_col_name = ''.join([col, old_col_suffix])
            if is_target and old_col_suffix == '(t)':
                output_data_frame[new_col_name] = output_data_frame[old_col_name]
                current_columns = list(output_data_frame.columns.values)
                current_columns.remove(new_col_name)
                current_columns.remove('Date')
                new_column_order = ['Date', new_col_name]
                new_column_order.extend(current_columns)
                output_data_frame = output_data_frame[new_column_order]
                output_data_frame.drop(columns=old_col_name, inplace=True)
            elif (is_target and old_col_suffix != '(t)') or (not is_target and re.search('t-1', old_col_suffix) is not None):
                output_data_frame[new_col_name] = output_data_frame[old_col_name]
                output_data_frame.drop(columns=old_col_name, inplace=True)

            else:
                output_data_frame.drop(columns=old_col_name, inplace=True)

ts_data_frame = output_data_frame.copy()
ts_data_frame.set_index('Date', inplace=True)
print(tabulate(ts_data_frame.tail(), ts_data_frame.columns.values))

history = pd.DataFrame(index=ts_data_frame.index.values)
history['Interval_Date'] = history.index.values
history['Interval_Date'] = history['Interval_Date'].shift(+INTERVAL_IN_DAYS)
history = pd.merge(history, mv_time_series,
                   how='left',
                   left_on='Interval_Date',
                   right_on='Date')
history.index = ts_data_frame.index.values
history = history[[TARGET_COL]]

groups = [3, 4, 5]
# if 1 in graphs:
#     i = 1
#     # plot each column
#     pyplot.figure()
#     for group in groups:
#         pyplot.subplot(len(groups), 1, i)
#         pyplot.plot(values[:, group])
#         pyplot.title(mv_time_series.columns[group], y=0.5, loc='right')
#         i += 1
#     pyplot.show()


groups = [3, 4, 5]
# if 3 in graphs:
#     i = 1
#     # plot each column
#     pyplot.figure()
#     for group in groups:
#         pyplot.subplot(len(groups), 1, i)
#         pyplot.plot(supervised_data2.values[:, group])
#         pyplot.title(supervised_data2.columns[group], y=0.5, loc='right')
#         i += 1
#     pyplot.show()

num_values = ts_data_frame.shape[0]
print(num_values, history.shape[0])
assert num_values == history.shape[0]
split_line = int(num_values*TRAIN_PCT)

train = ts_data_frame.values[:split_line, :]

test = ts_data_frame.values[split_line:, :]

historical_data = history.values[split_line:, :]

# split into input and outputs
train_X, train_y = train[:, 1:], train[:, 0]

test_X, test_y = test[:, 1:], test[:, 0]

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
# if 2 in graphs:
#     pyplot.plot(history.history['loss'], label='train')
#     pyplot.plot(history.history['val_loss'], label='test')
#     pyplot.legend()
#     pyplot.show()

# make a prediction
if VERBOSE:
    print("test_X.shape: %s sent to predict" % str(test_X.shape))

yhat = model.predict(test_X)

if VERBOSE:
    print("test_X.shape[2]: ", test_X.shape[2])
test_X = test_X.reshape((test_X.shape[0], test_X.shape[2]))


inv_yhat = pd.Series(inverse_difference(historical_data, yhat, INTERVAL_IN_DAYS).squeeze(axis=1))
# final_data = pd.merge(historical_data.tail(inv_yhat.shape[0]),
#                       inv_yhat,
#                       left_index=True,
#                       right_index=True,
#                       suffixes=('_actual', '_predicted'))

ts_test_data = mv_time_series.copy()
ts_test_data = ts_test_data.tail(inv_yhat.shape[0])
inv_yhat.index = ts_test_data.index.values

final_data = pd.merge(mv_time_series, pd.DataFrame(inv_yhat, columns=['Predicted Close']), left_index=True, right_index=True)
# exit()
# final_data['Predicted Close'] = final_data[dif_tgt_col_nm]
# final_data.drop(columns = dif_tgt_col_nm, inplace=True)
final_data['Pct Difference'] = (final_data['Predicted Close'] - final_data[TARGET_COL])*100 / final_data[TARGET_COL]
rmse = np.sqrt(mean_squared_error(mv_time_series[[TARGET_COL]].tail(inv_yhat.shape[0]).values, inv_yhat))
print('Test RMSE: %.3f' % rmse)

# length_of_test = test_y.shape[0]*-1
# previous_price = mv_time_series['Price'][length_of_test:]
print(tabulate(final_data.tail(15), final_data.columns.values))

print("Min Diff: %0.3f%s, Mean Diff: %0.3f%s, StdDev Diff: %0.3f%s, Max Diff: %0.3f%s" %
      (final_data['Pct Difference'].min(), "%",
       final_data['Pct Difference'].mean(), "%",
       final_data['Pct Difference'].std(), "%",
       final_data['Pct Difference'].max(), "%"))

if 4 in graphs:
    pyplot.plot(final_data[TARGET_COL], label='Actual')
    pyplot.plot(final_data['Predicted Close'], label='Predicted')
    pyplot.legend()
    pyplot.show()

if 5 in graphs:
    pyplot.plot(final_data['Pct Difference'], label='Percent Difference')
    pyplot.plot(final_data['VIX_Close'], label='VIX')
    pyplot.legend()
    pyplot.show()
