import math
import tensorflow as tf
from python_tf import input
from python_tf import configures as conf
from pathlib import Path
import numpy as np


def bidireonal_rnn(drop_out):
    """
    The bidireonal rnn model. The user characteristics is processed by another network and then added with the output
    of the output of the hidden layer. The result is the input of the output layer.
    :return: the predictions for all steps.
    """
    x_input = tf.keras.layers.Input(shape=[conf.NUM_DAYS, conf.NUM_BRANDS * conf.NUM_POS], name='x')
    user_characteristics_input = tf.keras.layers.Input(shape=[1], name='user_characteristics')
    brand_price_index_input = tf.keras.layers.Input(shape=[conf.NUM_DAYS, conf.NUM_BRANDS], name='brand_price_index')

    # # user attributes layers
    user_characteristics = tf.keras.layers.Dropout(rate=drop_out, name='user_characteristics_dropout')(user_characteristics_input)
    user_characteristics = tf.keras.layers.Dense(
        units=conf.NUM_DAYS * conf.NUM_BRANDS,
        activation='tanh',
        bias_initializer='zeros',
        name='user_characteristics_layer')(user_characteristics)
    user_characteristics = tf.keras.layers.Reshape(target_shape=[conf.NUM_DAYS, conf.NUM_BRANDS])(user_characteristics)

    # # rnn layers
    x = tf.keras.layers.Concatenate(axis=2, name='conc_x_and_brand_price_index')([x_input, brand_price_index_input])
    # x: [batch_size, num_days, 2 * num_lstm_state]
    x = tf.keras.layers.Bidirectional(
        layer=tf.keras.layers.LSTM(
            units=conf.NUM_LSTM_STATE,
            dropout=drop_out,
            return_sequences=True,
            input_shape=(conf.NUM_DAYS, conf.NUM_BRANDS * (conf.NUM_POS + 1))
        ),
        name='x_bi_lstm_rnn')(x)
    # x: [batch_size, num_days, num_brands]
    x = tf.keras.layers.Dense(
        units=conf.NUM_BRANDS,
        activation='tanh',
        kernel_initializer=tf.keras.initializers.truncated_normal(
            stddev=1.0 / math.sqrt(2 * conf.NUM_LSTM_STATE)),
        bias_initializer='zeros',
        name='x_out')(x)

    # # predictions
    predictions = tf.keras.layers.Add()([x, user_characteristics])
    predictions = tf.keras.layers.Dropout(rate=drop_out, name='output_dropout')(predictions)
    predictions = tf.keras.layers.Dense(
        units=conf.NUM_BRANDS,
        activation='sigmoid',
        kernel_initializer=tf.keras.initializers.truncated_normal(
            stddev=1.0 / math.sqrt(conf.NUM_BRANDS)),
        bias_initializer='zeros',
        name='output_layer')(predictions)

    # build the model
    model = tf.keras.Model(inputs=[x_input, user_characteristics_input, brand_price_index_input], outputs=predictions)

    return model


def compile_model(model, learning_rate, beta1=0.9, beta2=0.999, epsilon=1e-8):
    model.compile(optimizer=tf.train.AdamOptimizer(learning_rate=learning_rate,
                                                   beta1=beta1,
                                                   beta2=beta2,
                                                   epsilon=epsilon),
                  loss='binary_crossentropy',
                  metrics=['accuracy', 'binary_crossentropy', ])


if __name__ == '__main__':
    parrent_path = str(Path.cwd().parent)
    tfrecord_file_names = [parrent_path + '/' + conf.data_path]
    sess = tf.keras.backend.get_session()
    batch, ini = input.get_dataset(tfrecord_file_names, 2014, True)
    sess.run([ini, tf.global_variables_initializer(), tf.local_variables_initializer()])

    rnn_model = bidireonal_rnn(0.9)

    compile_model(rnn_model, 0.0001)

    x, u, b, y = sess.run([batch['x'], batch['user_characteristics'], batch['brand_price_index'], batch['y']])

    rnn_model.fit(x=[batch['x'], batch['user_characteristics'], batch['brand_price_index']],
              y=batch['y'], steps_per_epoch =1000, epochs=10000)

