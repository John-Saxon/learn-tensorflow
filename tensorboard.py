import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt


def add_layer(inputs, in_size, out_size, activation_function=None):
    # pass
    with tf.name_scope("layer"):
        with tf.name_scope("weights"):
            Weights = tf.Variable(tf.random_normal(
                [in_size, out_size]), name="W")
        with tf.name_scope("pian_zhi"):
            biases = tf.Variable(tf.zeros([1, out_size]))+0.1
        with tf.name_scope("Wx_plus_b"):
            Wx_plus_b = tf.matmul(inputs, Weights)+biases
        if activation_function is None:
            outputs = Wx_plus_b
        else:
            outputs = activation_function(Wx_plus_b)

        return outputs


x_data = np.linspace(-1, 1, 300, dtype=np.float32)[:, np.newaxis]
noise = np.random.normal(0, 0.05, x_data.shape).astype(np.float32)
y_data = np.square(x_data)-0.5+noise

with tf.name_scope("inputs"):
    xs = tf.placeholder(tf.float32, [None, 1], name="x_input")
    ys = tf.placeholder(tf.float32, [None, 1], name="y_input")

l1 = add_layer(xs, 1, 10, activation_function=tf.nn.relu)
prediction = add_layer(l1, 10, 1, activation_function=None)

with tf.name_scope("shun_shi"):
    loss = tf.reduce_mean(tf.reduce_sum(
        tf.square(ys-prediction), reduction_indices=[1]))

with tf.name_scope("xun_lian"):
    trian_step = tf.train.GradientDescentOptimizer(0.1).minimize(loss)

init = tf.global_variables_initializer()
s = tf.Session()

writer = tf.summary.FileWriter(
    "/Users/johnsaxon/test/github.com/learn-tensorflow/checkpoints/", s.graph)

s.run(init)

fig = plt.figure()
ax = fig.add_subplot(1, 1, 1)
ax.scatter(x_data, y_data, marker='x')
plt.ion()
plt.show()

for i in range(1000):
    s.run(trian_step, feed_dict={xs: x_data, ys: y_data})
    if i % 50 == 0:
        print(s.run(loss, feed_dict={xs: x_data, ys: y_data}))
        try:
            ax.lines.remove(lines[0])
        except Exception:
            pass
        prediction_value = s.run(prediction, feed_dict={xs: x_data})
        # ax.lines.remove(lines[0])
        lines = ax.plot(x_data, prediction_value, 'r-', lw=5)
        plt.pause(0.1)

plt.ioff()
plt.show()
