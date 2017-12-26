from __future__ import print_function
from __future__ import division

import warnings

import cv2
import numpy as np

from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler

from keras.utils import np_utils
from keras.preprocessing.image import flip_axis, transform_matrix_offset_center, apply_transform


def data_scaler(x, statistical_scaling=True, scaler=None):
    x_is_list = isinstance(x, list)
    if not x_is_list:
        x = [x]

    return_scaler = scaler is None
    if return_scaler:
        scaler = []

    for i in range(len(x)):
        x[i] = x[i].astype('float32')

        if statistical_scaling:
            orig_shape = x[i].shape
            x_flatten = x[i].reshape(len(x[i]), -1)
            if return_scaler:
                cur_scaler = StandardScaler()
                cur_scaler.fit(x_flatten)
            else:
                cur_scaler = scaler[i]

            x_flatten = cur_scaler.transform(x_flatten)
            x[i] = x_flatten.reshape(orig_shape)
        else:
            x[i] -= 128.
            x[i] /= 255

            cur_scaler = StandardScaler()

            cur_scaler.mean_ = np.ones(x[i].shape[1:], np.float32).flatten() * 128.
            cur_scaler.scale_ = np.full(x[i].shape[1:], 255, dtype=np.float32).flatten()

        if return_scaler:
            scaler.append(cur_scaler)

    if return_scaler:
        return x if x_is_list else x[0], scaler
    else:
        return x if x_is_list else x[0]


def preprocess_labels(labels_train, labels_validation, labels_test, categorical=True):
    def process_labels_single(l, enc, nb_classes):
        l_enc = enc.transform(l).astype(np.int32)
        if categorical:
            l_enc = np_utils.to_categorical(l_enc, nb_classes)
        return l_enc

    encoder = LabelEncoder()
    encoder.fit(labels_train)
    nb_classes = len(encoder.classes_)  # Avoids Edge-cases where validation/test sets miss label representation

    y_train = process_labels_single(labels_train, encoder, nb_classes)
    if labels_validation is not None:
        y_validation = process_labels_single(labels_validation, encoder, nb_classes)
    else:
        y_validation = None
    y_test = process_labels_single(labels_test, encoder, nb_classes)

    return y_train, y_validation, y_test, encoder


def rotate_augmentation(x, rotate_deg, row_index=1, col_index=2):
    rot_mat = cv2.getRotationMatrix2D((x.shape[col_index] / 2, x.shape[row_index] / 2), rotate_deg, 1.0)

    x_new = np.empty_like(x)
    for ind, img in enumerate(x):
        new_img = cv2.warpAffine(img, rot_mat, (x.shape[col_index], x.shape[row_index]),
                                 borderMode=cv2.BORDER_REPLICATE)
        if new_img.ndim == 2:  # Handle adding the channels axis to a single channel image
            new_img = np.expand_dims(new_img, 2)
        x_new[ind] = new_img

    # show_data(x_new, title="rotate_augmentation")
    return x_new


def crop_augmentation(x, crop_size):
    if x.ndim == 4:
        x_padded = np.pad(x, ((0, 0), (crop_size, crop_size), (crop_size, crop_size), (0, 0)), 'edge')
        x_cropped_top_left = x_padded[:, :-crop_size * 2, :-crop_size * 2, :]
        x_cropped_top_right = x_padded[:, :-crop_size * 2, crop_size * 2:, :]
        x_cropped_bottom_left = x_padded[:, crop_size * 2:, :-crop_size * 2, :]
        x_cropped_bottom_right = x_padded[:, crop_size * 2:, crop_size * 2:, :]
    else:
        x_padded = np.pad(x, ((crop_size, crop_size), (crop_size, crop_size), (0, 0)), 'edge')
        x_cropped_top_left = x_padded[:-crop_size * 2, :-crop_size * 2, :]
        x_cropped_top_right = x_padded[:-crop_size * 2, crop_size * 2:, :]
        x_cropped_bottom_left = x_padded[crop_size * 2:, :-crop_size * 2, :]
        x_cropped_bottom_right = x_padded[crop_size * 2:, crop_size * 2:, :]

    return np.concatenate((x_cropped_top_left, x_cropped_top_right, x_cropped_bottom_left, x_cropped_bottom_right))


def illuminate_augmentation(x, illumination):
    if x.shape[3] == 3:
        raise Exception("Color image illumination not implemented. Color-space couldn't be resolved (RGB/BGR/Lab etc.)")

        # # Code for BGR image illumination using Lab space
        # img = cv2.imread('temp.jpg')
        # cv2.imshow('Original', img)
        # cv2.waitKey(0)
        # lab_img = cv2.cvtColor(img, cv2.COLOR_BGR2Lab)
        # lab_img[..., 0] = cv2.add(lab_img[..., 0], illumination)
        # bgr_img_out = cv2.cvtColor(lab_img, cv2.COLOR_Lab2BGR)
        # cv2.imshow('lighter_out_img', bgr_img_out)
        # cv2.waitKey(0)

    x_new = cv2.add(x.astype(np.uint8), illumination)
    x_new = x_new.astype(np.float32)

    # show_data(x_new, title="illuminate_augmentation")
    return x_new


def scale_augmentation(x, scale):
    x_new = resize_batch(x, scale)
    dx = (x.shape[1] - x_new.shape[1]) // 2
    dy = (x.shape[2] - x_new.shape[2]) // 2
    x_padded = np.pad(x_new, ((0, 0), (dx, x.shape[1] - x_new.shape[1] - dx), (dy, x.shape[2] - x_new.shape[2] - dy),
                              (0, 0)), 'edge')
    return x_padded


def data_augmentation(x, labels, augmentation_params, col_index=2):
    """Notice that augmentation is currently not incremental"""
    x_is_list = isinstance(x, list)
    if not x_is_list:
        x = [x]

    rotate_deg = augmentation_params.rotate_deg

    x_augmented = []
    for xi in x:
        xi_augmented = xi
        if augmentation_params.flip_axis_horizontal:
            x_flipped = flip_axis(xi, axis=col_index)
            xi_augmented = np.concatenate((xi_augmented, x_flipped))

        for illumination in augmentation_params.illumination:
            if illumination:
                x_illuminated = illuminate_augmentation(xi, illumination)
                xi_augmented = np.concatenate((xi_augmented, x_illuminated))

        for rotate_deg in augmentation_params.rotate_deg:
            if rotate_deg:
                x_rotated = rotate_augmentation(xi, rotate_deg=rotate_deg)
                xi_augmented = np.concatenate((xi_augmented, x_rotated))

        for scale in augmentation_params.scale:
            if scale != 1:
                xi_scaled = scale_augmentation(xi, scale)
                xi_augmented = np.concatenate((xi_augmented, xi_scaled))

        for crop_size in augmentation_params.crop_size:
            if crop_size:
                xi_cropped = crop_augmentation(xi, crop_size)
                xi_augmented = np.concatenate((xi_augmented, xi_cropped))

        x_augmented.append(xi_augmented)

    labels_augmented = labels

    if augmentation_params.flip_axis_horizontal:
        flip_labels = augmentation_params.flip_labels
        if flip_labels is None:
            flip_labels = labels
        labels_augmented = np.concatenate((labels_augmented, flip_labels))

    for illumination in augmentation_params.illumination:
        if illumination:
            labels_augmented = np.concatenate((labels_augmented, labels))

    for rotate_deg in augmentation_params.rotate_deg:
        if rotate_deg:
            labels_augmented = np.concatenate((labels_augmented, labels))

    for scale in augmentation_params.scale:
        if scale != 1:
            labels_augmented = np.concatenate((labels_augmented, labels))

    # Crop should be last
    for crop_size in augmentation_params.crop_size:
        if crop_size:
            labels_augmented = np.concatenate((labels_augmented, np.tile(labels, (4, 1))))

    # show_data(x_augmented, title="data_augmentation")
    return x_augmented if x_is_list else x_augmented[0], labels_augmented


def sample_from_data(x, y, nb_samples, file_names=None):
    y = np.squeeze(y)

    if y.ndim == 1:
        y_type = 'num'
        nb_classes = np.max(y)+1
    elif y.ndim == 2:
        y_type = 'categorical'
        nb_classes = y.shape[1]
    else:
        raise Exception("Invalid y - should be categorical (2 dims array) or numerical (1 dim vector)")

    sampled_ind = []
    for class_id in range(nb_classes):
        if y_type == 'num':
            class_ind_bool = y == class_id
        elif y_type == 'categorical':
            class_ind_bool = y[:, class_id] == 1
        else:
            raise Exception("Invalid y type")

        class_ind = np.asarray(range(y.shape[0]))[class_ind_bool]
        class_sampled_ind = np.random.choice(class_ind, size=min(class_ind.shape[0], nb_samples), replace=False)

        sampled_ind.extend(class_sampled_ind)

    x = x[sampled_ind]
    y = y[sampled_ind]
    if file_names is not None:
        file_names = file_names[sampled_ind]

    if file_names is not None:
        return x, y, file_names
    else:
        return x, y


def resize_batch(x, resize):
    if resize >= 1.0:
        warnings.warn("You are enlarging images - not standard")

    img_num = x.shape[0]
    new_rows = np.round(x.shape[1] * resize).astype(np.uint)
    new_cols = np.round(x.shape[2] * resize).astype(np.uint)
    new_x = np.empty((img_num, new_rows, new_cols, x.shape[3]), dtype=np.float32)

    x = x.astype(np.float32)
    for i in range(img_num):
        new_img = cv2.resize(x[i], (new_cols, new_rows), interpolation=cv2.INTER_AREA)
        if new_img.ndim == 2:  # Handle adding the channels axis to a single channel image
            new_img = np.expand_dims(new_img, 2)
        new_x[i] = new_img

    return new_x


# FIXME: add support of list of x
def crop_batch(x, crop_rect):
    if not isinstance(crop_rect, tuple) or len(crop_rect) != 4:
        raise Exception("crop_rect must be tuple of - (x, y, width, height)")

    img_num = x.shape[0]
    new_rows = crop_rect[3]
    new_cols = crop_rect[2]
    if crop_rect[0] < 0 or crop_rect[0] > x.shape[2] or crop_rect[1] < 0 or crop_rect[1] > x.shape[1] \
        or new_cols < 0 or crop_rect[0] + new_cols > x.shape[2] \
            or new_rows < 0 or crop_rect[1] + new_rows > x.shape[1]:
        raise Exception("crop_rect exceed image dimensions {0} {1}".format(crop_rect, x.shape))

    new_x = np.empty((img_num, new_rows, new_cols, x.shape[3]), dtype=np.float32)
    for i in range(img_num):
        new_x[i] = x[i, crop_rect[1]:crop_rect[1]+new_rows, crop_rect[0]:crop_rect[0]+new_cols, :]

    return new_x


class AugmentationParams:
    def __init__(self, flip_axis_horizontal=False, flip_labels=None, rotate_deg=0, crop_size=0, illumination=0,
                 scale=1):
        self.flip_axis_horizontal = flip_axis_horizontal
        self.flip_labels = flip_labels

        if isinstance(rotate_deg, list):
            self.rotate_deg = rotate_deg
        else:
            self.rotate_deg = [rotate_deg]

        if isinstance(crop_size, list):
            self.crop_size = crop_size
        else:
            self.crop_size = [crop_size]

        if isinstance(illumination, list):
            self.illumination = illumination
        else:
            self.illumination = [illumination]

        if isinstance(scale, list):
            self.scale = scale
        else:
            self.scale = [scale]

    def __str__(self):
        return "Augmentation params: flip_axis_horizontal [" + str(self.flip_axis_horizontal) \
               + "], rotate_deg [" + str(self.rotate_deg) + "], crop_size [" + str(self.crop_size) +\
               "], illumination [" + str(self.illumination) + "], scale [" + str(self.scale) + "]"


class LabelsEncoderParams:
    def __init__(self, categorical=True):
        self.categorical = categorical

    def __str__(self):
        return "LabelsEncoder params: categorical [" + str(self.categorical) + "]"


class ScalerParams:
    def __init__(self, statistical=True, crop_rect=None, resize=1.0):
        self.statistical = statistical
        self.crop_rect = crop_rect  # tuple - (x, y, width, height)
        self.resize = resize

    def __str__(self):
        return "Scaler params: statistical [" + str(self.statistical) \
               + "], crop_rect [" + str(self.crop_rect) + "], resize [" + str(self.resize) + "]"


def preprocess(x_train, x_validation, x_test, labels_train, labels_validation, labels_test,
               encoder_params=None, augmentation_params=None, scaler_params=None, flatten=False, dim_ordering='tf'):
    print("Pre-processing data...")

    x_is_list = isinstance(x_train, list)
    if not x_is_list:
        x_train = [x_train]
        if x_validation is not None:
            x_validation = [x_validation]
        x_test = [x_test]

    if encoder_params is not None:
        print("Preparing labels...")
        print(encoder_params)
        y_train, y_validation, y_test, encoder = preprocess_labels(labels_train, labels_validation, labels_test,
                                                                   encoder_params.categorical)
        classes_num = len(encoder.classes_)
        print('Number of classes: %i' % classes_num)
    else:
        encoder = None
        classes_num = None
        y_train = labels_train
        y_validation = labels_validation
        y_test = labels_test

    if augmentation_params is not None:
        print("Augmenting data...")
        print(augmentation_params)
        x_train, y_train = data_augmentation(x_train, y_train, augmentation_params)

    if scaler_params is None:
        scaler_params = ScalerParams(statistical=False)

    print("Scaling data...")
    print(scaler_params)

    if scaler_params.crop_rect is not None:
        x_train = crop_batch(x_train, scaler_params.crop_rect)
        if x_validation is not None:
            x_validation = crop_batch(x_validation, scaler_params.crop_rect)
        x_test = crop_batch(x_test, scaler_params.crop_rect)

    if scaler_params.resize != 1.0:
        x_train = resize_batch(x_train, scaler_params.resize)
        if x_validation is not None:
            x_validation = resize_batch(x_validation, scaler_params.resize)
        x_test = resize_batch(x_test, scaler_params.resize)

    print("Scaling data ({0})...".format('Statistical scaling' if scaler_params.statistical else 'Only 255 div'))
    x_train, scaler = data_scaler(x_train, statistical_scaling=scaler_params.statistical)
    if x_validation is not None:
        x_validation = data_scaler(x_validation, statistical_scaling=scaler_params.statistical, scaler=scaler)
    x_test = data_scaler(x_test, statistical_scaling=scaler_params.statistical, scaler=scaler)

    if dim_ordering == 'th':
        x_train = [np.transpose(xi, (0, 3, 1, 2)) for xi in x_train]
        if x_validation is not None:
            x_validation = [np.transpose(xi, (0, 3, 1, 2)) for xi in x_validation]
        x_test = [np.transpose(xi, (0, 3, 1, 2)) for xi in x_test]

    image_shape = []
    for i in range(len(x_train)):
        image_shape.append(x_train[i].shape[1:])

        if flatten:
            x_train[i] = x_train[i].reshape(len(x_train[i]), -1)
            if x_validation is not None:
                x_validation[i] = x_validation[i].reshape(len(x_validation[i]), -1)
            x_test[i] = x_test[i].reshape(len(x_test[i]), -1)

        print('x_train shape:', x_train[i].shape)
        print(x_train[i].shape[0], 'train samples')
        if x_validation is not None:
            print(x_validation[i].shape[0], 'validation samples')
        print(x_test[i].shape[0], 'test samples')
        print('Image shape: %s' % (image_shape,))
        print("")

    if not x_is_list:
        x_train = x_train[0]
        if x_validation is not None:
            x_validation = x_validation[0]
        x_test = x_test[0]
        image_shape = image_shape[0]
        scaler = scaler[0]

    if encoder_params is not None:
        return x_train, x_validation, x_test, labels_train, labels_validation, labels_test, y_train, y_validation,\
               y_test, encoder, scaler, image_shape, classes_num
    else:
        return x_train, x_validation, x_test, labels_train, labels_validation, labels_test, y_train, y_validation,\
               y_test, scaler, image_shape
