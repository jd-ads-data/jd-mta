import math
import tensorflow as tf
from tf import input
from tf import metrics
from pathlib import Path
import generate_simulation_data as gsd

NUM_BRANDS = 2
NUM_DAYS = 15
NUM_POS = 10
NUM_LSTM_STATE = 64
LEARNING_DECAY_STEP_SIZE = 1000
LEARNING_RATE = 0.0001
THRESHOLD_FOR_CLASSIFICATION= 0.5
LEARNING_RATE_DECAY_FACTOR = 0.9


def _variable_on_cpu(name, shape, initializer):
    """Helper to create a Variable stored on CPU memory.

    Args:
      name: name of the variable
      shape: list of ints
      initializer: initializer for Variable

    Returns:
      Variable Tensor
    """
    with tf.device('/cpu:0'):
        dtype = tf.float32
        var = tf.get_variable(name, shape, initializer=initializer, dtype=dtype)
    return var


def get_a_variable_on_cpu(name, shape, normal_init=True):
    if normal_init:
        initializer = tf.truncated_normal_initializer(stddev=1.0 / math.sqrt(float(shape[-1])), dtype=tf.float32)
    else:
        initializer = tf.zeros_initializer

    return _variable_on_cpu(name, shape, initializer)


def bidireonal_rnn(x, brand_profile, user_profile, keep_prob):
    """
    The bidireonal rnn model. The user profile is processed by another network and then added with the output
    of the output of the hidden layer. The result is the input of the output layer.
    :param x: [batch_size, num_days, num_brand * num_pos]
    :param brand_profile: [batch_size, num_days, num_brand]
    :param user_profile: [1]
    :return: the predions for all steps and the predions for the last step.
    """

    with tf.variable_scope('user_profile_layer'):
        weights_user_profile = get_a_variable_on_cpu(
            name='weights_user_profile', shape=[1, NUM_BRANDS], normal_init=True)
        bias_user_profile = get_a_variable_on_cpu(
            name='bias_user_profile', shape=[NUM_BRANDS], normal_init=False)
        # user_profile: [batch_size, num_brands]
        user_profile = tf.matmul(user_profile, weights_user_profile) + bias_user_profile
        # user_profile: [batch_size, num_brands * num_days]
        user_profile = tf.tile(user_profile, [1, NUM_DAYS])
        user_profile = tf.nn.tanh(user_profile)
        user_profile = tf.nn.dropout(user_profile, keep_prob=keep_prob)
        # user_profile: [batch_size * num_days, num_brands]
        user_profile = tf.reshape(user_profile, [-1, NUM_BRANDS])

    with tf.name_scope('lstm_layer'):
        x = tf.concat([x, brand_profile], 2)
        drop_out = 1 - keep_prob
        # output_fw: [batch_size, num_days, num_lstm_state]
        # output_bw: [batch_size, num_days, num_lstm_state]
        x = \
        tf.keras.layers.Bidirectional(tf.keras.layers.LSTM(units=NUM_LSTM_STATE,
                           dropout=drop_out,
                           return_sequences=True,
                           input_shape=(NUM_DAYS, NUM_BRANDS * 11)
                           ))(x)
        # outputs: [batch_size * num_days, num_lstm_state]
        outputs = tf.reshape(x, [-1, 2 * NUM_LSTM_STATE])
        weights_lstm_out = get_a_variable_on_cpu(
            name='weights_lstm_out',
            shape=[2 * NUM_LSTM_STATE, NUM_BRANDS],
            normal_init=True)
        # outputs: [batch_size * num_days, num_brands]
        outputs = tf.matmul(outputs, weights_lstm_out)

    with tf.name_scope('output_layer'):
        # predictions: [batch_size * num_days, num_brands]
        predictions = outputs + user_profile
        # predictions: [batch_size * num_days, num_brands]
        predictions = tf.nn.dropout(predictions, keep_prob=keep_prob)
        # predions: [batch_size, num_days, num_brands]
        predictions = tf.reshape(predictions, [-1, NUM_DAYS, NUM_BRANDS])
        # predictionslast_step: [batch_size, num_brands]
        predictions_last_step = tf.reshape(predictions[:, NUM_DAYS - 1, :], [-1, NUM_BRANDS])

    return predictions, predictions_last_step


def loss(predictions, labels):
    with tf.variable_scope(name_or_scope='losses') as scope:
        # Calculate the cross entropy as the loss.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions, name='sigmoid_cross_entropy')
        loss = tf.reduce_mean(loss)
    return loss


def get_loss(scope, y, x, brand_profile, user_profile, keep_prob):
    # Build a Graph.
    # Get the model
    model = bidireonal_rnn
    logits_predictions, _ = model(x, brand_profile, user_profile, keep_prob)
    cross_entropy_loss = loss(predictions=logits_predictions, labels=y)

    # Add some summaries and metrics.
    validation_metrics = metrics.get_validation_metrics(
        logits_predictions=logits_predictions, labels=y, threshold=THRESHOLD_FOR_CLASSIFICATION, scope=scope)
    return cross_entropy_loss, validation_metrics


def train(total_loss, global_step):
    # Variables that affect learning rate.
    decay_steps = LEARNING_DECAY_STEP_SIZE

    # Decay the learning rate exponentially based on the number of steps.
    lr = tf.train.exponential_decay(LEARNING_RATE,
                                    global_step,
                                    decay_steps,
                                    LEARNING_RATE_DECAY_FACTOR,
                                    staircase=True)

    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES)
    summaries.append(tf.summary.scalar(name='learning_rate', tensor=lr))

    # Compute gradients.
    opt = tf.train.AdamOptimizer(lr)
    grads = opt.compute_gradients(total_loss)

    # Apply gradients.
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    with tf.control_dependencies([apply_gradient_op]):
        train_op = tf.no_op(name='train')

    return train_op


if __name__ == '__main__':
    parrent_path = str(Path.cwd().parent)
    tfrecord_file_names = [parrent_path + '/' + gsd.data_path]
    sess = tf.Session()
    iterator, output_types, output_shapes = input.get_dataset_iterator(tfrecord_file_names, 128, True)
    batch = input.get_dataset_batch(output_types, output_shapes)
    string_handle = sess.run(iterator.string_handle())

    cross_entropy_loss, validation_metrics = get_loss(
        scope=None, y=batch['y'], x=batch['x'], brand_profile=batch['brand_profile'], user_profile=batch['user_profile'], keep_prob=0.8)

    global_step = tf.get_variable(
        'global_step', [],
        initializer=tf.constant_initializer(0), trainable=False)

    train_op = train(cross_entropy_loss, global_step)

    sess.run(tf.global_variables_initializer())

    for i in range(10000):
        a = sess.run([cross_entropy_loss, train_op], feed_dict={input.handle: string_handle})
        if i % 100 == 0:
            print(a[0])
