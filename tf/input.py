import tensorflow as tf
import generate_simulation_data as gsd
from pathlib import Path

handle = tf.placeholder(tf.string, shape=[])


def _parse_function(example_proto):
    sequence_features = {
        'x': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'user_profile': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'brand_profile': tf.FixedLenSequenceFeature([], dtype=tf.float32),
        'y': tf.FixedLenSequenceFeature([], dtype=tf.float32),
    }
    _, sequence_parsed = tf.parse_single_sequence_example(
        serialized=example_proto,
        sequence_features=sequence_features
    )

    x = sequence_parsed['x']
    user_profile = sequence_parsed['user_profile']
    brand_profile = sequence_parsed['brand_profile']
    y = sequence_parsed['y']
    input_tensors = [x, user_profile, brand_profile, y]

    return input_tensors


def get_dataset_iterator(file_list, batch_size, shuffle=False):
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(
        lambda x, user_profile, brand_profile, y:
        {
            # x: [num_days, num_brand * num_pos]
            'x': tf.reshape(x, [gsd.NUM_DAYS, gsd.NUM_BRANDS * gsd.NUM_POS]),
            # user_profile: [1]
            'user_profile': tf.reshape(user_profile, [1]),
            # brand_profile: [num_days, num_brands]
            'brand_profile': tf.reshape(brand_profile, [gsd.NUM_DAYS, gsd.NUM_BRANDS]),
            # y: [num_days, num_brands]
            'y': tf.reshape(y, [gsd.NUM_DAYS, gsd.NUM_BRANDS]),
        })
    if shuffle:
        dataset = dataset.shuffle(buffer_size=batch_size * 2)
    batched_dataset = dataset.batch(batch_size)
    batched_dataset = batched_dataset.repeat()
    iterator = batched_dataset.make_one_shot_iterator()
    output_types = batched_dataset.output_types
    output_shapes = batched_dataset.output_shapes
    return iterator, output_types, output_shapes


def get_dataset_batch(output_types, output_shapes):
    iterator = tf.data.Iterator.from_string_handle(
        handle, output_types, output_shapes)
    next_batch = iterator.get_next()
    return next_batch


if __name__ == '__main__':
    parrent_path = str(Path.cwd().parent)
    tfrecord_file_names = [parrent_path + '/' + gsd.data_path]
    sess = tf.Session()
    iterator, training_types, training_shapes = get_dataset_iterator(
        file_list=tfrecord_file_names,
        batch_size=1,
    )
    batch = get_dataset_batch(training_types, training_shapes)
    training_handle = sess.run(iterator.string_handle())

    for i in range(15):
        a = tf.reduce_sum(batch['x'])
        print(sess.run(a, feed_dict={handle: training_handle}))
