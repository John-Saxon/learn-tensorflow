import tensorflow as tf

state = tf.Variable(0, name="counter")
print(state)
print(state.name)
print(state.shape)
print(state.dtype)


one = tf.constant(1)

new_value = tf.add(state, one)
print(new_value)
update = tf.assign(state, new_value)
print(update)

# deprecated after 2017-03-02
# init = tf.initialize_all_variables()  # must have if define variable

init = tf.global_variables_initializer()

with tf.Session() as s:
    s.run(init)
    for _ in range(3):
        s.run(update)
        print(s.run(state))
        print(state)
