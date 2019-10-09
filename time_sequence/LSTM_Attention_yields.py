
# https://www.cnblogs.com/mtcnn/p/9411597.html
# https://machinelearningmastery.com/convert-time-series-supervised-learning-problem-python/

import os
import glob
from sklearn.preprocessing import MinMaxScaler
import sys
import seaborn as sns
import pandas as pd
import numpy as np
# from tensorflow import keras
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, LSTM, Dropout
# from keras.models import Sequential
# from keras.layers import Dense, LSTM, Dropout
import matplotlib.pyplot as plt
# % matplotlib inline  # only valid in jupyter

columns = ['YEAR', 'MONTH', 'DAY', 'TEMP_HIG',
           'TEMP_COL', 'AVG_TEMP', 'AVG_WET', 'DATA_COL']
data = pd.read_csv(
    '/Users/johnsaxon/test/github.com/learn-tensorflow/data/industry_timeseries/timeseries_train_data/1.csv', names=columns)

# print(data.head())
# print(data.shape)

plt.figure(figsize=(24, 8))
for i in range(8):
    plt.subplot(8, 1, i+1)
    plt.plot(data.values[:, i])
    plt.title(columns[i], y=0.5, loc='right')
# plt.show()


def series_to_supervised(data, n_in=1, n_out=1, dropnan=True):
    n_vars = 1 if type(data) is list else data.shape[1]
    df = pd.DataFrame(data)
    cols, names = list(), list()
    # input sequence (t-n, ... t-1)
    for i in range(n_in, 0, -1):
        cols.append(df.shift(1))
        names += [('var%d(t-%d)' % (j+1, i)) for j in range(n_vars)]
    # print(cols.head())

    # forecast sequence (t, t+1, ... t+n)
    for i in range(0, n_out):
        cols.append(df.shift(-i))
        if i == 0:
            names += [('var%d(t)' % (j+1)) for j in range(n_vars)]
        else:
            names += [('var%d(t+%d)' % (j+1, i)) for j in range(n_vars)]
    # print(cols.head())
    # put it all together
    agg = pd.concat(cols, axis=1)
    agg.columns = names

    # drop rows with NaN values
    if dropnan:
        agg.dropna(inplace=True)
    # print(agg.head())
    return agg


# normalization
scaler = MinMaxScaler(feature_range=(0, 1))
scaled_data = scaler.fit_transform(
    data[['DATA_COL', 'TEMP_HIG', 'TEMP_COL', 'AVG_TEMP', 'AVG_WET']].values)

# trans time-series to supervised
reframed = series_to_supervised(scaled_data, 1, 1)

# drop useless data
reframed.drop(reframed.columns[[6, 7, 8, 9]], axis=1, inplace=True)
print(reframed.info())
print(reframed.head())


# train data
train_days = 400
valid_days = 150
values = reframed.values

train = values[:train_days, :]
valid = values[train_days:train_days+valid_days, :]
test = values[train_days+valid_days:, :]

train_X, train_y = train[:, :-1], train[:, -1]
valid_X, valid_y = valid[:, :-1], valid[:, -1]
test_X, test_y = test[:, :-1], test[:, -1]

# reconstruct data
train_X = train_X.reshape((train_X.shape[0], 1, train_X.shape[1]))
valid_X = valid_X.reshape((valid_X.shape[0], 1, valid_X.shape[1]))
test_X = test_X.reshape((test_X.shape[0], 1, test_X.shape[1]))
print(train_X.shape, train_y.shape, valid_X.shape,
      valid_y.shape, test_X.shape, test_y.shape)


model = Sequential()
model.add(LSTM(50, activation='relu', input_shape=(
    train_X.shape[1], train_X.shape[2]), return_sequences=True))
model.add(Dense(1, activation='linear'))
model.compile(loss='mean_squared_error', optimizer='adam')

# training
LSTM = model.fit(train_X, train_y, epochs=100,
                 validation_data=(valid_X, valid_y),
                 batch_size=32, verbose=2, shuffle=False)

# plot history
plt.plot(LSTM.LSTM['loss'], label='train')
plt.plot(LSTM.LSTM['val_loss'], label='valid')
plt.legend()
plt.show()


# prediction and visualization
plt.figure(figsize=(24, 8))
train_predict = model.predict(train_X)
valid_predict = model.predict(valid_X)
test_predict = model.predict(test_X)

plt.plot(values[:, -1], c='b')
plt.plot([x for x in train_predict], c='g')
plt.plot([None for _ in train_predict]+[x for x in valid_predict], c='y')
plt.plot([None for _ in train_predict] +
         [None for _ in valid_predict] + [x for x in test_predict], c='r')
plt.show()
