import tensorflow as tf
import numpy as np
import keras

NUM_LSTM_STATE = 10
drop_out = 0.8

sess = tf.Session()

a = np.zeros([2, 3])
b = tf.zeros([2, 4, 4])
a[0, 1] = 2


d = keras.layers.Bidirectional(keras.layers.LSTM(units=NUM_LSTM_STATE,
                                                       dropout=drop_out,
                                                       return_sequences=True,
                                                       input_shape=(2, 4, 4)))(b)
sess.run([tf.global_variables_initializer(), tf.local_variables_initializer()])

print(sess.run(d))

m = np.zeros([2,3,4])

print(m[0])
print(m[0, 1])
