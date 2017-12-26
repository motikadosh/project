from __future__ import print_function
from __future__ import division

import numpy as np
from sklearn.model_selection import train_test_split


def get_split_indexes(y, validation_percent=0.1, test_percent=0.2, preserve_class_sampling=False):
    # type: (np.ndarray, float, float, bool) -> np.ndarray, np.ndarray, np.ndarray
    full_ind = range(y.shape[0])

    stratify_y = None
    if preserve_class_sampling:
        stratify_y = y

    ind_train, ind_test, y_train, y_test = train_test_split(full_ind, y, test_size=test_percent, stratify=stratify_y)

    if validation_percent > 0:
        stratify_y_train = None
        if preserve_class_sampling:
            stratify_y_train = y_train

        train_ind = range(y_train.shape[0])
        ind_ind_train, ind_ind_val, y_train, y_validation =\
            train_test_split(train_ind, y_train, test_size=validation_percent, stratify=stratify_y_train)
        ind_train = np.array(ind_train)
        ind_val = ind_train[ind_ind_val]
        ind_train = ind_train[ind_ind_train]
    else:
        ind_val = None

    return ind_train, ind_val, ind_test


def split_data(x, y, validation_percent=0.1, test_percent=0.2, preserve_class_sampling=False, file_names=None):
    """Split arrays or matrices into random train, test and validation subsets
    Notes:
        The validation percentage is calculated from the training set ONLY.
        The train and test percentage must sum up to 1.
    """
    if validation_percent < 0 or validation_percent >= 1:
        raise Exception("Invalid validation_percent argument")

    if test_percent < 0 or test_percent >= 1:
        raise Exception("Invalid test_percent argument")

    ind_train, ind_validation, ind_test = get_split_indexes(y, validation_percent, test_percent,
                                                            preserve_class_sampling)
    x_train = x[ind_train]
    y_train = y[ind_train]
    x_test = x[ind_test]
    y_test = y[ind_test]

    if ind_validation is not None:
        x_validation = x[ind_validation]
        y_validation = y[ind_validation]
    else:
        x_validation = None
        y_validation = None

    if file_names is not None:
        file_names_train = file_names[ind_train]
        file_names_test = file_names[ind_test]
        if ind_validation is not None:
            file_names_val = file_names[ind_validation]
        else:
            file_names_val = None

        return x_train, x_validation, x_test, y_train, y_validation, y_test, file_names_train, file_names_val, \
            file_names_test
    else:
        return x_train, x_validation, x_test, y_train, y_validation, y_test

