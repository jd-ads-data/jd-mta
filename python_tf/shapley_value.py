#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import math

from python_tf import configures as conf


def get_bit(num, index):
    return num & (1 << index) != 0


def count_1_in_binary(n):
    return bin(n).count("1")


def compute_shapley_value(x, user_profile, brand_profile, brand_index, predict_fn):
    """
    Compute the Shapley value of each tuple of (time, ad_position) in one brand for a conversion by the exact method,
    and then sum all the Shapley values for each ad position across all days.
    :param x: ad exposure sequence, [conf.NUM_DAYS, num_brand * conf.NUM_POS]
    :param user_profile: the user profile for the user [1]
    :param brand_profile: the brand index for each brand, [conf.NUM_DAYS, num_brand]
    :param brand_index: the index of the brand, []
    :param predict_fn: a method which predicts the probability of the conversion in the last day given the features
    :return: the Shapley value for each tuple of (time, ad_position) of the brand, [conf.NUM_DAYS, conf.NUM_POS]
    """
    shapley_value_for_tuple = np.zeros([conf.NUM_DAYS, conf.NUM_POS])
    non_zero_tuple_index_list = []
    for i in range(conf.NUM_DAYS):
        for j in range(0, conf.NUM_POS):
            if x[i, j + brand_index * conf.NUM_POS] > 1.0e-6:
                non_zero_tuple_index_list.append((i, j))

    num_tuples = len(non_zero_tuple_index_list)
    num_cases = 2 ** num_tuples
    for case_index in range(num_cases):
        cp_x = np.copy(x)
        for j in range(num_tuples):
            if not get_bit(case_index, j):
                date_index = non_zero_tuple_index_list[j][0]
                brand_pos_index = non_zero_tuple_index_list[j][1] + brand_index * conf.NUM_POS
                cp_x[date_index, brand_pos_index] = 0.0

        prediction = predict_fn(cp_x, user_profile, brand_profile, brand_index)

        num_pos_in_case_index = count_1_in_binary(case_index)
        factorial_n = math.factorial(num_tuples)
        for j in range(num_tuples):
            if get_bit(case_index, j):
                s = num_pos_in_case_index - 1
                value_to_add = 1. * math.factorial(num_tuples - s - 1) * math.factorial(s) / factorial_n * prediction
                shapley_value_for_tuple[non_zero_tuple_index_list[j]] += value_to_add
            else:
                s = num_pos_in_case_index
                value_to_minus = 1. * math.factorial(num_tuples - s - 1) * math.factorial(s) / factorial_n * prediction
                shapley_value_for_tuple[non_zero_tuple_index_list[j]] -= value_to_minus
        # print(shapley_value_for_tuple)

    # normalize the shapley value
    x_0 = np.copy(x)
    for tup in non_zero_tuple_index_list:
        date_index = tup[0]
        brand_pos_index = tup[1] + brand_index * conf.NUM_POS
        x_0[date_index, brand_pos_index] = 0.0
        if shapley_value_for_tuple[tup] < 0.0:
            shapley_value_for_tuple[tup] = 0.0

    p_x = predict_fn(x, user_profile, brand_profile, brand_index)
    p_0 = predict_fn(x_0, user_profile, brand_profile, brand_index)
    inc_p = np.max([p_x - p_0, 0])
    sum_all_shapley_values = np.max([np.sum(shapley_value_for_tuple), 0.0])

    if sum_all_shapley_values > 1e-10 and inc_p > 1.0e-10:
        for tup in non_zero_tuple_index_list:
            shapley_value_for_tuple[tup] = shapley_value_for_tuple[tup] * inc_p / sum_all_shapley_values
    else:
        shapley_value_for_tuple = np.zeros([conf.NUM_DAYS, conf.NUM_POS])

    shapley_value_for_pos = np.sum(shapley_value_for_tuple, 1)

    return shapley_value_for_pos


if __name__ == '__main__':
    print('Hi')
