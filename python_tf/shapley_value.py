#!/usr/bin/env python
# encoding: utf-8

import numpy as np
import math


def get_bit(num, index):
    return num & (1 << index) != 0


def compute_shapley_value(x, brand_profile, user_profile, brand_index, predict_fn):
    """
    Compute the Shapley value of each tuple of (time, ad_position) in one brand for a conversion by the exact method.
    :param x: ad exposure sequence, [num_days, num_brand * num_pos]
    :param brand_profile: the brand index for each brand, [num_days, num_brand]
    :param user_profile: the user profile for the user [1]
    :param brand_index: the index of the brand, []
    :param predict_fn: a method which predicts the probability of the conversion in the last day given the features
    :return: the Shapley value for each tuple of (time, ad_position) of the brand, [num_days, num_pos]
    """
    num_days, num_brands, num_pos = np.shape(x)
    shapley_value_for_tuple = np.zeros([num_days, num_pos])
    non_zero_tuple_index_list = []
    for i in range(num_days):
        for j in range(brand_index * num_pos, (brand_index + 1) * num_pos):
            if x[i, j] > 1e-6:
                non_zero_tuple_index_list.append((i, j))

    num_tuples = len(non_zero_tuple_index_list)
    num_cases = 2 ** num_tuples
    for case_index in range(num_cases):
        cp_x = np.copy(x)
        for j in range(num_tuples):
            if not get_bit(case_index, j):
                cp_x[non_zero_tuple_index_list[j]] = 0.0

        prediction = predict_fn(cp_x, user_profile, brand_profile, brand_index)

        for j in range(num_tuples):
            if get_bit(case_index, j):
                s = num_tuples - 1
                value_to_add = 1. * math.factorial(num_tuples - s - 1) * math.factorial(s) * prediction
                shapley_value_for_tuple[non_zero_tuple_index_list[j]] += value_to_add
            else:
                s = num_tuples - 1
                value_to_minus = 1. * math.factorial(num_tuples - s - 1) * math.factorial(s) * prediction
                shapley_value_for_tuple[non_zero_tuple_index_list[j]] -= value_to_minus

    return shapley_value_for_tuple


if __name__ == '__main__':
    print('Hi')
