import os
import errno
import pickle
import shutil

from contextlib import contextmanager
from datetime import datetime

import numpy as np

import consts


def get_timestamp():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


class SessionInfo:
    def __init__(self, title, out_dir=None):
        self.title = title
        if out_dir is None:
            self.out_dir = self.title + '_' + get_timestamp()
        else:
            self.out_dir = out_dir


def mkdirs(full_path):
    """Create all directories in the given path if they does not exist"""

    if not os.path.exists(full_path):
        try:
            os.makedirs(full_path)
        except OSError as exc:  # Guard against race condition
            if exc.errno != errno.EEXIST:
                raise


@contextmanager
def cd(new_dir):
    prev_dir = os.getcwd()
    os.chdir(os.path.expanduser(new_dir))
    try:
        yield
    finally:
        os.chdir(prev_dir)


def load_pickle(pickle_full_path):
    with open(pickle_full_path) as f:
        print("Loading history from pickle [" + pickle_full_path + "]")
        data = pickle.load(f)
        return data


def save_pickle(sess_info, data):
    pickle_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, 'pickle')
    mkdirs(pickle_dir)

    pickle_fname = '{sess_title}.pkl'.format(sess_title=sess_info.title)
    pickle_full_path = os.path.join(pickle_dir, pickle_fname)

    with open(pickle_full_path, "wb") as f:
        print("Saving history to pickle")
        pickle.dump(data, f)


def get_files_with_ext(base_dir, ext_list=None):
    if ext_list is None:
        ext_list = ('.jpg', '.jpeg', '.gif', '.png')

    matches = []
    for root, dirs, files in os.walk(base_dir):
        for filename in files:
            if filename.endswith(ext_list):
                matches.append(os.path.join(root, filename))
    return matches


# TODO: Use get_files_with_ext to count the files
def count_files_in_folder(path, extentions):
    if type(extentions) is str:
        extentions = [extentions]
    elif type(extentions) is not list:
        raise Exception("Argument is not a list or string")

    if not os.path.isdir(path):
        raise Exception(path + " does not exist or is not a directory")

    # print("Counting files in: " + path)
    count = 0
    for root, dirs, files in os.walk(path):
        for name in files:
            for ext in extentions:
                if os.path.splitext(name)[1] == ext:
                    # print(os.path.join(root, name))
                    count += 1
    # print("Found {0} files in {1}".format(count, path))
    return count


def parse_shape(shape):
    from keras import backend as keras_backend

    dim_order = keras_backend.image_dim_ordering()
    if dim_order == 'th':
        channels, rows, cols = shape
    elif dim_order == 'tf':
        rows, cols, channels = shape
    else:
        raise Exception("Unexpected dim order")

    return rows, cols, channels


def get_image_from_batch(x, n):
    from keras import backend as keras_backend

    image = x[n]

    dim_order = keras_backend.image_dim_ordering()
    if dim_order == 'th':
        return np.transpose(image, (1, 2, 0))
    elif dim_order == 'tf':
        return image
    else:
        raise Exception("Unexpected dim order")
