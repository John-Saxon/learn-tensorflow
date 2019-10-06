import tensorflow as tf
import numpy as np

a = np.arange(1, 10)
print(a)


t = tf.reshape(a, [3, 3])
with tf.Session() as s:
    print(s.run(t))
