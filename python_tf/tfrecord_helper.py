import tensorflow as tf
import numpy as np


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


def bytes_sequence_feature(values):
    feature_list = [
        _bytes_feature(value) for value in values
    ]
    return tf.train.FeatureList(feature=feature_list)


def int64_sequence_feature(values):
    feature_list = [
        _int64_feature(value) for value in values
    ]
    return tf.train.FeatureList(feature=feature_list)


def float_sequence_feature(values):
    feature_list = [
        _float_feature(value) for value in values
    ]
    return tf.train.FeatureList(feature=feature_list)
