#!/usr/bin/env python
# encoding: utf-8

import tensorflow as tf


def true_positives(predictions, labels, threshold=0.5):
    predictions = tf.cast(predictions, tf.float32)
    predictions = tf.cast(tf.greater_equal(x=predictions, y=threshold), tf.bool)
    labels = tf.cast(labels, tf.bool)
    num_true_positives = tf.count_nonzero(tf.cast(tf.logical_and(x=labels, y=predictions), tf.int32))
    return tf.cast(num_true_positives, tf.float32)


def true_negatives(predictions, labels, threshold=0.5):
    predictions = tf.cast(predictions, tf.float32)
    predictions = tf.cast(tf.greater_equal(x=predictions, y=threshold), tf.bool)
    labels = tf.cast(labels, tf.bool)
    num_true_negatives = tf.count_nonzero(tf.cast(tf.logical_not(tf.logical_or(x=labels, y=predictions)), tf.int32))
    return tf.cast(num_true_negatives, tf.float32)


def false_positives(predictions, labels, threshold=0.5):
    predictions = tf.cast(predictions, tf.float32)
    predictions = tf.cast(tf.greater_equal(x=predictions, y=threshold), tf.int32)
    labels = tf.cast(labels, tf.int32)
    num_false_positives = tf.count_nonzero(tf.cast(tf.greater(x=predictions, y=labels), tf.int32))
    return tf.cast(num_false_positives, tf.float32)


def false_negatives(predictions, labels, threshold=0.5):
    predictions = tf.cast(predictions, tf.float32)
    predictions = tf.cast(tf.greater_equal(x=predictions, y=threshold), tf.int32)
    labels = tf.cast(labels, tf.int32)
    num_false_negatives = tf.count_nonzero(tf.cast(tf.greater(x=labels, y=predictions), tf.int32))
    return tf.cast(num_false_negatives, tf.float32)


def accuracy_t(tp, tn, fp, fn):
    return 1.0 * (tp + tn) / (tp + tn + fp + fn)


def accuracy(predictions, labels, threshold=0.5):
    tp = true_positives(predictions, labels, threshold)
    tn = true_negatives(predictions, labels, threshold)
    fp = false_positives(predictions, labels, threshold)
    fn = false_negatives(predictions, labels, threshold)
    return accuracy_t(tp=tp, tn=tn, fp=fp, fn=fn)


def recall_t(tp, fn):
    return 1.0 * tp / (tp + fn)


def recall(predictions, labels, threshold=0.5):
    tp = true_positives(predictions, labels, threshold)
    fn = false_negatives(predictions, labels, threshold)

    return recall_t(tp=tp, fn=fn)


def precision_t(tp, fp):
    return 1.0 * tp / (tp + fp)


def precision(predictions, labels, threshold=0.5):
    tp = true_positives(predictions, labels, threshold)
    fp = false_positives(predictions, labels, threshold)

    return precision_t(tp=tp, fp=fp)


def loss(predictions, labels):
    with tf.variable_scope(name_or_scope='losses') as scope:
        # Calculate the cross entropy as the loss.
        loss = tf.nn.sigmoid_cross_entropy_with_logits(labels=labels, logits=predictions, name='sigmoid_cross_entropy')
        loss = tf.reduce_mean(loss)
    return loss


def get_validation_metrics(logits_predictions, labels, threshold, scope=None):
    class ValidationMetrics(object):
        pass

    predictions = tf.sigmoid(logits_predictions)
    validation_loss = loss(predictions=logits_predictions, labels=labels)
    tp = true_positives(predictions=predictions, labels=labels, threshold=threshold)
    tn = true_negatives(predictions=predictions, labels=labels, threshold=threshold)
    fp = false_positives(predictions=predictions, labels=labels, threshold=threshold)
    fn = false_negatives(predictions=predictions, labels=labels, threshold=threshold)
    recall_value = recall_t(tp=tp, fn=fn)
    precision_value = precision_t(tp=tp, fp=fp)
    accuracy_value = accuracy_t(tp=tp, tn=tn, fp=fp, fn=fn)

    validation_metrics = ValidationMetrics()
    validation_metrics.predictions = predictions
    validation_metrics.loss = validation_loss
    validation_metrics.recall = recall_value
    validation_metrics.precision = precision_value
    validation_metrics.accuracy = accuracy_value
    validation_metrics.tp = tp
    validation_metrics.tn = tn
    validation_metrics.fp = fp
    validation_metrics.fn = fn

    # Add validation summaries.
    summaries = tf.get_collection(tf.GraphKeys.SUMMARIES, scope=scope)
    summaries.append(tf.summary.histogram('predicted_values', predictions))
    summaries.append(tf.summary.scalar('validation_loss', validation_loss))
    summaries.append(tf.summary.scalar('validation_recall', recall_value))
    summaries.append(tf.summary.scalar('validation_precision', precision_value))
    summaries.append(tf.summary.scalar('validation_accuracy', accuracy_value))
    summaries.append(tf.summary.scalar('validation_tp', tp))
    summaries.append(tf.summary.scalar('validation_fp', fp))
    summaries.append(tf.summary.scalar('validation_tn', tn))
    summaries.append(tf.summary.scalar('validation_fn', fn))

    return validation_metrics
