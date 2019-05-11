import tensorflow as tf
import numpy as np
import keras

NUM_LSTM_STATE = 10
drop_out = 0.8

sess = tf.Session()

b = tf.zeros([2, 2, 2])
c = tf.ones([2,2,2])
# c[0,0,0] = 0.5

a = tf.keras.layers.Add()([b, c])
# a = tf.keras.layers.Dense(units=3)(a)

sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

print(sess.run(tf.shape(a)))
k = sess.run(a)
print(k)

