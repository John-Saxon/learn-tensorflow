import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

tf.set_random_seed(1)


# 导入数据
minst = input_data.read_data_sets('MNIST_data', one_hot=True)

# hyperparameters

lr = 0.001  # learning rate
training_iters = 100000
batch_size = 128
n_inputs = 28
n_steps = 28  # time steps
n_hidden_units = 128  # neurons in hidden layer
n_classes = 10

x = tf.placeholder(tf.float32, [None, n_steps, n_inputs])
y = tf.placeholder(tf.float32, [None, n_classes])

# weights biases init
weights = {
    # shape (28,128)
    'in': tf.Variable(tf.random_normal([n_inputs, n_hidden_units])),
    # shape (128,10)
    'out': tf.Variable(tf.random_normal([n_hidden_units, n_classes]))
}

biases = {
    # shape(128,)
    'in': tf.Variable(tf.constant(0.1, shape=[n_hidden_units, ])),
    # shape(10, )
    'out': tf.Variable(tf.constant(0.1, shape=[n_classes, ]))
}


def RNN(X, weights, biases):
    # X==>(128 batches * 28 steps, 28 inputs)
    X = tf.reshape(X, [-1, n_inputs])

    # X_in  = W*X + b
    X_in = tf.matmul(X, weights['in']) + biases['in']

    # X_in ==> (128 batches, 28 steps, 128 hidden) trans back to 3-dimenson
    X_in = tf.reshape(X_in, [-1, n_steps, n_hidden_units])

    lstm_cell = tf.contrib.rnn.BasicLSTMCell(
        n_hidden_units, forget_bias=1.0, state_is_tuple=True)

    init_state = lstm_cell.zero_state(batch_size, dtype=tf.float32)

    outputs, final_state = tf.nn.dynamic_rnn(
        lstm_cell, X_in, initial_state=init_state, time_major=False)

    # Methods 1
    results = tf.matmul(final_state[1], weights['out']) + biases['out']

    # Methods 2
    # outputs = tf.unstack(tf.transpose(outputs, [1, 0, 2]))
    # results = tf.matmul(outputs[-1], weights['out']) + biases['out']

    return results


pred = RNN(x, weights, biases)

# def softmax_cross_entropy_with_logits_v2(
#     _sentinel=None,  # pylint: disable=invalid-name
#     labels=None,
#     logits=None,
#     dim=-1,
#     name=None)

cost = tf.reduce_mean(
    tf.nn.softmax_cross_entropy_with_logits_v2(labels=y, logits=pred))
train_op = tf.train.AdamOptimizer(lr).minimize(cost)


correct_pred = tf.equal(tf.argmax(pred, 1), tf.argmax(y, 1))
accuracy = tf.reduce_mean(tf.cast(correct_pred, tf.float32))


init = tf.global_variables_initializer()

with tf.Session() as s:
    s.run(init)
    step = 0
    while step*batch_size < training_iters:
        batch_xs, batch_ys = minst.train.next_batch(batch_size)
        batch_xs = batch_xs.reshape([batch_size, n_steps, n_inputs])
        s.run([train_op], feed_dict={
            x: batch_xs,
            y: batch_ys,
        })

        if step % 20 == 0:
            print(s.run(accuracy, feed_dict={
                x: batch_xs,
                y: batch_ys,
            }))
        step += 1
