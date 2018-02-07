from __future__ import print_function
from __future__ import division

import os
import errno
import pickle
import shutil
import random

from contextlib import contextmanager
from datetime import datetime

import numpy as np
from pyquaternion import Quaternion
import cv2
from tqdm import tqdm

import consts


def get_timestamp():
    return datetime.now().strftime("%Y_%m_%d-%H_%M_%S")


class SessionInfo:
    def __init__(self, title, out_dir=None, postfix=None):
        self.title = title

        self.out_dir = (self.title + '_' + get_timestamp()) if out_dir is None else out_dir
        if postfix is not None:
            self.out_dir += postfix


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


def get_image_size(img):
    """Size is (width, height), Shape (Assume Tensorflow/OpenCV style) is (row, cols, channels)"""
    return img.shape[1], img.shape[0]

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


def get_files_with_ext(base_dir, ext_list=None, recursive=True, abs_path=False, sort=False, warn_empty=True):
    if ext_list is None:
        ext_list = ('.jpg', '.jpeg', '.gif', '.png')

    print("Getting files from [%s], ext list [%s], recursive [%s], abs_path [%s], sort [%s]" %
          (base_dir, ext_list, recursive, abs_path, sort))

    matches = []
    if recursive:
        for root, dirs, files in os.walk(base_dir):
            for filename in files:
                if filename.endswith(ext_list):
                    matches.append(os.path.join(root, filename))
    else:
        for filename in os.listdir(base_dir):
            if filename.endswith(ext_list):
                matches.append(os.path.join(base_dir, filename))

    if not abs_path:
        n = len(base_dir + os.path.sep)
        matches[:] = [match[n:] for match in matches]

    if sort:
        matches = sorted(matches)

    if warn_empty and len(matches) == 0:
        import warnings
        warnings.warn("Path [" + base_dir + "] had 0 results!")

    return matches


def part_of(*arrays, **options):
    if len(arrays) == 0:
        raise ValueError("At least one array required as input")

    part = options.pop('part', 0.0)
    if not part:
        raise Exception("Argument 'part' missing")

    import numbers
    if part <= 0 or (part > 1.0 and not isinstance(part, numbers.Integral)):
        raise Exception("Invalid part value [%s] or type [%s]" % (part, type(part)))

    res_arrays = []
    for array in arrays:
        if part <= 1.0:
            cur_part = int(len(array) * part)
        else:
            cur_part = part

        res_arrays.append(array[:cur_part])

    return tuple(res_arrays) if len(arrays) > 1 else res_arrays[0]


def load_folder(cache_prefix, folder, image_size, ext_list=None, part_of_data=1.0, shuffle=False, sort_by_name=False,
                recursive=False, save_cache=True, skip_upscale=False, labels_parser=None, process_image_fn=None):
    print("Loading folder [%s], image_size [%s], part_of_data [%s], sort_by_name [%s], recursive [%s], save_cache [%s]"
          % (folder, image_size, part_of_data, sort_by_name, recursive, save_cache))

    # db_path = os.path.join(consts.LOCAL_DATABASES_DIR, folder)
    db_path = folder
    if not os.path.isdir(db_path):
        raise Exception("folder " + folder + " does not exist")

    with cd(db_path):
        labels = None

        params_prefix = cache_prefix + '_' + str(part_of_data)
        params_prefix += "_%sx%s" % (image_size[0], image_size[1]) if image_size is not None else ""
        params_prefix += "_r" if recursive else ""
        print("Cache params_prefix:", params_prefix)

        if image_size is not None and os.path.isfile(params_prefix + "_x.npy") or \
                image_size is None and os.path.isfile(params_prefix + "_file_urls.npy"):

            print("Using cache...")
            if image_size is not None:
                x = np.load(params_prefix + "_x.npy")
            file_urls = np.load(params_prefix + "_file_urls.npy")
            if os.path.isfile(params_prefix + "_labels.npy"):
                labels = np.load(params_prefix + "_labels.npy")
            # FIXME: Add loading of multiple output models
        else:
            print("Loading files...")
            files = get_files_with_ext(db_path, ext_list=ext_list, recursive=recursive, sort=sort_by_name)

            if shuffle:
                print("Shuffling [%d] files before load..." % len(files))
                random.shuffle(files)

            files = part_of(files, part=part_of_data)

            img_num = len(files)
            if labels_parser is not None:
                labels = labels_parser.allocate_labels(img_num)

            file_urls = np.empty(img_num, dtype="object")

            if image_size is not None:
                x = np.empty((img_num, image_size[1], image_size[0], 1), dtype=np.float32)

            cnt = 0
            label = None
            for cur_file in tqdm(files):
                if image_size is not None:
                    # TODO: Support color images
                    img = cv2.imread(cur_file, cv2.IMREAD_GRAYSCALE)
                    if img is None:
                        continue

                    if img.shape[1] / float(img.shape[0]) != image_size[0] / float(image_size[1]):
                        print("Wrong aspect ratio. Expected %f, received %f" % (image_size[1] / float(image_size[0]),
                                                                                img.shape[1] / float(img.shape[0])))
                        continue

                    if skip_upscale and (img.shape[0] < image_size[0] or img.shape[1] < image_size[1]):
                        print("Skipping small image. Expected (%s,%s), received (%s,%s)" %
                              (image_size[0], image_size[1], img.shape[0], img.shape[1]))
                        continue

                    img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)

                    if process_image_fn is not None:
                        img = process_image_fn.process(img, cur_file, label)
                        if img is None:
                            continue

                    x[cnt, :, :, 0] = img

                if labels_parser is not None:
                    label = labels_parser.read_label(cur_file)
                    if label is None:
                        print(cur_file + " label is not valid skipping")
                        continue

                    if isinstance(labels, list):
                        for i in range(len(labels)):
                            labels[i][cnt] = label[i]
                    else:
                        labels[cnt] = label

                file_urls[cnt] = os.path.join(folder, cur_file)

                cnt += 1

            if image_size is not None:
                x = x[:cnt]

            if labels_parser is not None:
                if isinstance(labels, list):
                    for i in range(len(labels)):
                        labels[i] = labels[i][:cnt]
                else:
                    labels = labels[:cnt]
            file_urls = file_urls[:cnt]

            if save_cache:
                print("Cache saved")
                if image_size is not None:
                    np.save(params_prefix + "_x.npy", x)
                if labels is not None:
                    if isinstance(labels, list):
                        for i in range(len(labels)):
                            np.save(params_prefix + "_labels_" + str(i) + ".npy", labels[i])
                    else:
                        np.save(params_prefix + "_labels.npy", labels)

                np.save(params_prefix + "_file_urls.npy", file_urls)

    print("Loaded " + str(file_urls.shape[0]) + " images from " + folder)
    return (labels, file_urls) if image_size is None else (x, labels, file_urls)


def min_max_scale(x, old_range, new_range):
        if x.dtype != np.float32:
            raise Exception("Invalid x dtype [%s]. Should be np.float32" % x.dtype)

        old_span = float(old_range[1]) - float(old_range[0])
        new_span = float(new_range[1]) - float(new_range[0])

        if old_span == 0 or new_span == 0:
            import warnings
            warnings.warn('old_span == 0 or new_span == 0')
            # raise Exception("Invalid old/new range")

        # This handles degenerate cases (Usually in debug)- E.g. Single (x, y) Many angles
        if old_span == 0:
            old_span = old_range[0]
        if new_span == 0:
            new_span = new_range[0]

        new_x = (((x - float(old_range[0])) * new_span) / old_span) + float(new_range[0])
        return new_x


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


# Currently not used - Converting rotation matrix to euler angles:
# https://www.learnopencv.com/rotation-matrix-to-euler-angles/
# OR-
# http://nghiaho.com/?page_id=846
# OR-
# http://planning.cs.uiuc.edu/node103.html
def get_yaw_pitch_roll_from_quat(quaternion):
    """ This according to:
    1) My notation of yaw-pitch-roll (Yaw- Left/Right, Pitch- Up/Down, Roll- Camera rotate).
    2) Camera pitch is normalized to look horizontally instead of down in the model. I.e its 90 degrees rotated.
       See: main.cpp @ getXfFromPose() -
            ...
            // Base rotation so we will be looking horizontall
            xyzXf = getCamRotMatDeg(0.0, -90, 0.0) * xyzXf;
            ..."""

    if type(quaternion) is np.ndarray:
        quaternion = Quaternion(quaternion)

    yay_pitch_roll = np.rad2deg(quaternion.yaw_pitch_roll)
    return np.array([yay_pitch_roll[0], yay_pitch_roll[2] + 90, yay_pitch_roll[1]])


def get_yaw_pitch_from_quaternion_array(arr):
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)
    return np.array([get_yaw_pitch_roll_from_quat(quat)[:2] for quat in arr])


def convert_quaternion_y_to_yaw_pitch(arr):
    if arr.ndim == 1:
        arr = np.expand_dims(arr, axis=0)

    if arr.shape[1] != 6:
        raise Exception("Input array must have 6 columns")

    return np.concatenate((arr[:, :2], get_yaw_pitch_from_quaternion_array(arr[:, 2:])), axis=-1).squeeze()


def quaternions_error(y, y_pred, normalized=False):
    if normalized:
        raise Exception("Normalized quaternions error not supported")

    y = get_yaw_pitch_from_quaternion_array(y)
    y_pred = get_yaw_pitch_from_quaternion_array(y_pred)

    return angle_l2_err(y, y_pred, normalized=normalized)


def rotation_error(y, y_pred, normalized=False):

    if y.shape[1] == 2:   # 'angle'
        return angle_l2_err(y, y_pred, normalized=normalized)
    elif y.shape[1] == 4:  # 'quaternion'
        return quaternions_error(y, y_pred, normalized=normalized)
    elif y.shape[1] == 9:   # 'matrix'
        pass
    else:
        raise Exception("Unknown y shape:", y.shape)