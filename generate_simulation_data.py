import numpy as np
import math
import tensorflow as tf

NUM_BRANDS = 2
NUM_DAYS = 15
NUM_POS = 10

AVG_NUM_IMP = 2.0

NUM_SAMPLES = 10000

lambda1 = 0.5
alpha0 = 1.0
alpha1 = np.abs(np.random.lognormal(0.0, 1.0, [NUM_BRANDS, NUM_POS]))
alpha2 = np.random.lognormal(0.0, 1.0, [NUM_BRANDS, 1])
b0 = -2000
b1 = np.random.uniform(0.1, 0.9, [NUM_BRANDS , NUM_POS])
b2 = [0.01, 0.01]
b3 = np.random.lognormal(0.0, 1.0, NUM_BRANDS)
brand_profile = np.abs(np.random.lognormal(0.0, 1.0, [NUM_DAYS, NUM_BRANDS]))

data_path = 'data/simulation_data.tfrecord'


def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    else:
        return 1 / (1 + math.exp(-gamma))


def generate_one_user():
    x = np.random.poisson(AVG_NUM_IMP, [NUM_DAYS, NUM_BRANDS, NUM_POS])
    user_profile = np.random.uniform(0.0, 1.0)

    h = np.zeros([NUM_DAYS + 1, 2])
    u = np.zeros([NUM_DAYS, 2])
    p = np.zeros([NUM_DAYS, 2])
    h[0, 0] = np.random.normal()
    h[1, 0] = np.random.normal()

    for b in range(0,2):
        for t in range(0, NUM_DAYS):
            h[t + 1, b] = lambda1 * sigmoid(alpha0 + np.dot(x[t, b], alpha1[b]) + \
                                            np.dot(brand_profile[t, b], alpha2[b])) + (1.0 - lambda1) * h[t, b]
            u[t, b] = b0 * user_profile + np.dot(x[t, b], b1[b]) + h[t + 1, b] * b2[b] + np.dot(brand_profile[t, b], b3[b])
            p[t, b] = sigmoid(u[t, b])

    y = np.random.binomial(np.ones(NUM_DAYS * NUM_BRANDS, dtype=int), np.reshape(p, NUM_DAYS * NUM_BRANDS))

    return x, user_profile, y



def _bytes_feature(value):
    if isinstance(value, (np.ndarray, list)):
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=value))
    else:
        return tf.train.Feature(bytes_list=tf.train.BytesList(value=[value]))


def _int64_feature(value):
    if isinstance(value, (np.ndarray, list)):
        return tf.train.Feature(int64_list=tf.train.Int64List(value=value))
    else:
        return tf.train.Feature(int64_list=tf.train.Int64List(value=[value]))


def _float_feature(value):
    if isinstance(value, (np.ndarray, list)):
        return tf.train.Feature(float_list=tf.train.FloatList(value=value))
    else:
        return tf.train.Feature(float_list=tf.train.FloatList(value=[value]))


def _bytes_sequence_feature(values):
    feature_list = [
        _bytes_feature(value) for value in values
    ]
    return tf.train.FeatureList(feature=feature_list)


def _int64_sequence_feature(values):
    feature_list = [
        _int64_feature(value) for value in values
    ]
    return tf.train.FeatureList(feature=feature_list)


def _float_sequence_feature(values):
    feature_list = [
        _float_feature(value) for value in values
    ]
    return tf.train.FeatureList(feature=feature_list)


def simulate_data_and_save(num_samples):
    writer = tf.python_io.TFRecordWriter(data_path)
    for i in range(num_samples):
        x, user_profile, y = generate_one_user()
        example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list={
                'x': _float_sequence_feature(np.reshape(x, [-1])),
                'user_profile': _float_sequence_feature(np.reshape(user_profile, [-1])),
                'brand_profile': _float_sequence_feature(np.reshape(brand_profile, [-1])),
                'y': _float_sequence_feature(np.reshape(y, [-1])),
            }))
        serialized = example.SerializeToString()
        writer.write(serialized)
    writer.close()


if __name__ == '__main__':
    simulate_data_and_save(1000)


