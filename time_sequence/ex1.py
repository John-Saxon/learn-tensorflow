import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf


# https://yq.aliyun.com/articles/118726

np.random.seed(111)

rng = pd.date_range(start='2000', periods=209, freq='M')
ts = pd.Series(np.random.uniform(-10, 10, size=len(rng)), rng).cumsum()

ts.plot(c='b', title='Example Time Series')
# plt.ion()
# plt.show()
plt.savefig("./test.png")
print(ts.head(10))


TS = np.array(ts)
num_periods = 20
f_horizon = 1

x_data = TS[:(len(TS)-(len(TS) % num_periods))]
print(x_data.shape)
x_batches = x_data.reshape(-1, 20, 1)

y_data = TS[1:(len(TS)-(len(TS) % num_periods))+f_horizon]
y_batches = y_data.reshape(-1, 20, 1)

print(len(x_batches))
print(x_batches.shape)
print(x_batches[0:2])

print(y_batches[0:1])
print(y_batches.shape)


def test_data(series, forcast, num_periods):
    test_x_setup = TS[-(num_periods+forcast):]
    testX = test_x_setup[:num_periods].reshape(-1, 20, 1)
    testY = TS[-(num_periods):].reshape(-1, 20, 1)
    return testX, testY


X_test, Y_test = test_data(TS, f_horizon, num_periods)
print(X_test.shape)
print(X_test)

tf.reset_default_graph()

inputs = 1
hidden = 100
output = 1
X = tf.placeholder(tf.float32, [None, num_periods, inputs])
y = tf.placeholder(tf.float32, [None, num_periods, output])

basic_cell = tf.contrib.rnn.BasicRNNCell(
    num_units=hidden, activation=tf.nn.relu)
rnn_output, states = tf.nn.dynamic_rnn(basic_cell, X, dtype=tf.float32)

lr = 0.001

stacked_rnn_output = tf.reshape(rnn_output, [-1, hidden])
stacked_outputs = tf.layers.dense(stacked_rnn_output, output)
outputs = tf.reshape(stacked_outputs, [-1, num_periods, output])

loss = tf.reduce_sum(tf.square(outputs-y))
optimizer = tf.train.AdamOptimizer(lr)
training_op = optimizer.minimize(loss)

init = tf.global_variables_initializer()

epochs = 1000

with tf.Session() as s:
    s.run(init)
    for ep in range(epochs):
        s.run(training_op, feed_dict={X: x_batches, y: y_batches})
        if ep % 100 == 0:
            mse = loss.eval(feed_dict={X: x_batches, y: y_batches})
            print(ep, "\tMSE:", mse)

        y_pred = s.run(outputs, feed_dict={X: X_test})
        print("-------------yyyyyyyyyyyyyyyyyy----------")
        print(y_pred)
