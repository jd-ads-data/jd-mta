import tensorflow as tf

from python_tf import configures as conf
from python_tf import input
from python_tf import model
from python_tf import shapley_value
import generate_simulation_data as gsd


def mta_example():
    # prepare the data
    print('------- Generating data.')
    training_file = 'data/training_data.tfrecord'
    evaluation_file = 'data/evaluation_data.tfrecord'
    gsd.simulate_data_and_save(10000, training_file)
    gsd.simulate_data_and_save(10000, evaluation_file)
    print('------- Save data successfully.')

    # get the input dataset and initialize the iterators
    print('------- Getting input datasets and initializing.')
    training_data, training_ini = input.get_dataset([training_file], 256, shuffle=True)
    evaluation_data, evaluation_ini = input.get_dataset([evaluation_file], 128)
    sess = tf.keras.backend.get_session()
    sess.run([training_ini, evaluation_ini])
    print('------- Dataset iterators initialized.')

    # build a model
    print('------- Build a model.')
    rnn_model = model.bidireonal_rnn(drop_out=0.75)

    # compile the model
    print('------- Compile the model.')
    model.compile_model(model=rnn_model, learning_rate=conf.LEARNING_RATE)

    # train the model
    print('------- Train the model using training data.')
    rnn_model.fit(
        x=[training_data['x'], training_data['user_characteristics'], training_data['brand_price_index']], y=training_data['y'],
        steps_per_epoch=2000, epochs=2)

    # evaluate the model
    print('------- Evaluate the model using evaluation data.')
    rnn_model.evaluate(x=[evaluation_data['x'], evaluation_data['user_characteristics'], evaluation_data['brand_price_index']],
                       y=evaluation_data['y'],
                       steps=100)

    # Compute the Shapley value for the data of which the y is 1.0
    def predict_fn(x, user_characteristics, brand_price_index, brand_index):
        prediction_all_step = rnn_model.predict(x=[[x], [user_characteristics], [brand_price_index]], batch_size=1)
        prediction = prediction_all_step[0, conf.NUM_DAYS - 1, brand_index]
        return prediction

    attribution_data, attribution_ini = input.get_dataset([training_file], 1)
    sess.run(attribution_ini)
    for i in range(1000):
        x_in, user_characteristics_in, brand_price_index_in, y_in = sess.run([
            attribution_data['x'], attribution_data['user_characteristics'],
            attribution_data['brand_price_index'], attribution_data['y']
        ])
        for brand_index in range(conf.NUM_BRANDS):
            if y_in[0, conf.NUM_DAYS - 1, brand_index] > 0.0:
                attribution_result = shapley_value.compute_shapley_value(
                    x_in[0], user_characteristics_in[0],
                    brand_price_index_in[0], brand_index, predict_fn)
                print('User %d, brand %d, Shapley value for each ad position: ' %(i, brand_index), '')
                print(attribution_result)


if __name__ == '__main__':
    mta_example()
