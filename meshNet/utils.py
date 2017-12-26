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


def part_of(part, *arrays):
    """This function assumes all arrays have identical length."""

    if len(arrays) == 0:
        raise Exception("No arrays passed")

    if part <= 0:
        raise Exception("Part argument cannot be <= 0")

    print("Handling part of data...")
    res_arrays = []
    for array in arrays:
        cur_part = part
        if cur_part <= 1.0:
            cur_part = int(len(array) * cur_part)
        res_arrays.append(array[:cur_part])

    return tuple(res_arrays) if len(arrays) > 1 else res_arrays[0]


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


def rm_file(file_path, skip_dirs=True):
    try:
        if os.path.isfile(file_path):
            os.unlink(file_path)
        elif not skip_dirs and os.path.isdir(file_path):
            shutil.rmtree(file_path)
    except Exception as e:
        print(e)


# import os, shutil
def rm_files(base_dir):
    for cur_file in os.listdir(base_dir):
        file_path = os.path.join(base_dir, cur_file)
        rm_file(file_path)


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


def xy_dist(y, y_pred):
    if y.shape[1] != 2:
        raise Exception("Unexpected dim - should be points array")

    dist = np.linalg.norm(y - y_pred, axis=-1)
    return dist


def angle_diff(a, b):
    # Return a signed difference between two angles
    # I.e. the minimum distance from src(a) to dst(b) - consider counter-clockwise as positive
    return (a - b + 180) % 360 - 180


def angle_l2_err(y, y_pred, normalized=False):
    f = 360.0 if normalized else 1.0
    return np.linalg.norm(angle_diff(y * f, y_pred * f) / f, axis=-1)

