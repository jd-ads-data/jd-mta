import numpy as np
import math
import tensorflow as tf

from python_tf import configures as conf
from python_tf import tfrecord_helper

AVG_NUM_IMP = 2.0
lambda1 = 0.5
alpha0 = 1.0
alpha1 = np.abs(np.random.lognormal(0.0, 1.0, [conf.NUM_BRANDS, conf.NUM_POS]))
alpha2 = np.random.lognormal(0.0, 1.0, [conf.NUM_BRANDS, 1])
beta0 = -200
beta1 = np.random.uniform(0.1, 0.9, [conf.NUM_BRANDS, conf.NUM_POS])
beta2 = 0.01
beta3 = np.random.lognormal(0.0, 1.0, conf.NUM_BRANDS)
brand_profile = np.random.lognormal(0.0, 0.25, [conf.NUM_DAYS, conf.NUM_BRANDS])

sum_p = 0.0
num_p = 0.0


def sigmoid(gamma):
    if gamma < 0:
        return 1 - 1 / (1 + math.exp(gamma))
    else:
        return 1 / (1 + math.exp(-gamma))


def generate_one_user():
    global sum_p, num_p
    x = np.random.poisson(AVG_NUM_IMP, [conf.NUM_DAYS, conf.NUM_BRANDS, conf.NUM_POS])
    user_profile = np.random.uniform(0.0, 1.0)

    h = np.zeros([conf.NUM_DAYS + 1, 2])
    u = np.zeros([conf.NUM_DAYS, 2])
    p = np.zeros([conf.NUM_DAYS, 2])
    h[0, 0] = np.random.normal()
    h[1, 0] = np.random.normal()

    for b in range(0, conf.NUM_BRANDS):
        for t in range(0, conf.NUM_DAYS):
            h[t + 1, b] = lambda1 * sigmoid(alpha0 + np.dot(x[t, b], alpha1[b]) + \
                                            np.dot(brand_profile[t, b], alpha2[b])) + (1.0 - lambda1) * h[t, b]
            u[t, b] = beta0 * user_profile + np.dot(x[t, b], beta1[b]) + h[t + 1, b] * beta2 + np.dot(brand_profile[t, b],
                                                                                                beta3[b])
            p[t, b] = sigmoid(u[t, b])

    y = np.random.binomial(np.ones(conf.NUM_DAYS * conf.NUM_BRANDS, dtype=int),
                           np.reshape(p, conf.NUM_DAYS * conf.NUM_BRANDS))

    sum_p += np.sum(p)
    num_p += p.size
    return x, user_profile, y


def simulate_data_and_save(num_samples, path):
    writer = tf.python_io.TFRecordWriter(path)
    for i in range(num_samples):
        x, user_profile, y = generate_one_user()
        example = tf.train.SequenceExample(
            feature_lists=tf.train.FeatureLists(feature_list={
                'x': tfrecord_helper.float_sequence_feature(np.reshape(x, [-1])),
                'user_profile': tfrecord_helper.float_sequence_feature(np.reshape(user_profile, [-1])),
                'brand_profile': tfrecord_helper.float_sequence_feature(np.reshape(brand_profile, [-1])),
                'y': tfrecord_helper.float_sequence_feature(np.reshape(y, [-1])),
            }))
        serialized = example.SerializeToString()
        writer.write(serialized)
    print('avg purchase rate is ', sum_p / num_p)
    writer.close()


if __name__ == '__main__':
    simulate_data_and_save(100000, conf.data_path)
