import pandas as pd

data = pd.read_csv('../data/SP500_VIX_20000101_20180919_3.csv',
                             header=0,
                             index_col=0,
                             thousands=',')

# Test type of column.values

print(type(data['SP500_Close'].values))
print("Returns ndarray")

print(type(data['SP500_Close']))
print("Returns Series")

print(type(data[['SP500_Close']]))
print("Returns 1 column DataFrame")

