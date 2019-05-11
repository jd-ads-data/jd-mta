import tensorflow as tf
from python_tf import configures as conf
from pathlib import Path


def _parse_function(example_proto):
    sequence_features = {
        'x': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'user_characteristics': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'brand_price_index': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
        'y': tf.io.FixedLenSequenceFeature([], dtype=tf.float32),
    }
    _, sequence_parsed = tf.io.parse_single_sequence_example(
        serialized=example_proto,
        sequence_features=sequence_features
    )

    x = sequence_parsed['x']
    user_characteristics = sequence_parsed['user_characteristics']
    brand_price_index = sequence_parsed['brand_price_index']
    y = sequence_parsed['y']
    input_tensors = [x, user_characteristics, brand_price_index, y]

    return input_tensors


def get_dataset(file_list, batch_size, shuffle=False, repeat=True):
    dataset = tf.data.TFRecordDataset(file_list)
    dataset = dataset.map(_parse_function)
    dataset = dataset.map(
        lambda x, user_characteristics, brand_price_index, y:
        {
            # x: [num_days, num_brands * num_pos]
            'x': tf.reshape(x, [conf.NUM_DAYS, conf.NUM_BRANDS * conf.NUM_POS]),
            # user_characteristics: [1]
            'user_characteristics': tf.reshape(user_characteristics, [1]),
            # brand_price_index: [num_days, num_brands]
            'brand_price_index': tf.reshape(brand_price_index, [conf.NUM_DAYS, conf.NUM_BRANDS]),
            # y: [num_days, num_brands]
            'y': tf.reshape(y, [conf.NUM_DAYS, conf.NUM_BRANDS]),
        })
    if shuffle:
        dataset = dataset.shuffle(buffer_size=int(batch_size * 1.5))
    batched_dataset = dataset.batch(batch_size)
    if repeat:
        batched_dataset = batched_dataset.repeat()
    iterator = batched_dataset.make_initializable_iterator()
    initializer = iterator.initializer
    batched_dataset = iterator.get_next()
    return batched_dataset, initializer


if __name__ == '__main__':
    parrent_path = str(Path.cwd().parent)
    tfrecord_file_names = [parrent_path + '/' + conf.data_path]
    sess = tf.Session()

    batch, ite = get_dataset(
        file_list=tfrecord_file_names,
        batch_size=1,
    )

    sess.run(ite)

    for i in range(15):
        a = tf.reduce_sum(batch['x'])
        print(sess.run(a, ))
