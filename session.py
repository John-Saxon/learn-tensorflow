import tensorflow as tf

m1 = tf.constant([[3, 3]])
m2 = tf.constant([[2], [2]])

product = tf.multiply(m1, m2)  # matrix multiply

p2 = tf.matmul(m1, m2)

# method 1
sess = tf.Session()

result = sess.run(product)
print(result)
print(sess.run(m1))
print(sess.run(m2))
sess.close()

with tf.Session() as s:
    print(s.run(p2))
