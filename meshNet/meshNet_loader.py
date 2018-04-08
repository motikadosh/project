from __future__ import print_function
from __future__ import division

import os
import pickle
import numpy as np
import cv2
from pyquaternion import Quaternion
from sklearn.utils import shuffle as sk_shuffle

import utils
import split
import consts
# from visualize import imshow

IMAGE_SIZE = (int(800/5), int(600/5))
# IMAGE_SIZE = (int(800/5), int((600/3)/5))


def apply_smoothing(image, kernel_size=7):
    """
    kernel_size must be positive and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


def flip_imgs_colors(imgs):
    # Flip image colors (White <-> Black)
    white_imgs = np.full_like(imgs, 255)
    imgs = white_imgs - imgs
    imgs[imgs > 0] = 255  # Make sure all values are 0/255
    return imgs


def load_image(file_url, image_size):
    img = cv2.imread(file_url, cv2.IMREAD_GRAYSCALE)
    if img.shape != image_size:
        # print("Resizing image on load: " + file_url + ", original size  " + str(img.shape))
        img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)

    return img


def load_pose_file(file_url):
    pose_file_url = os.path.splitext(file_url)[0][:-len('_edges')] + '.txt'
    with open(pose_file_url) as f:
        line = f.readline()
        _, pose_txt = line.split('=', 1)
        pose_txt = pose_txt.strip(' \t\n\r)(')
        pose = pose_txt.split(", ")
        if len(pose) != 6:
            raise Exception("Bad pose value in " + pose_file_url)

        pose = np.array([float(i) for i in pose])
        return pose


def load_xf_file(file_url):
    xf_file_url = os.path.splitext(file_url)[0][:-len('_edges')] + '.xf'
    with open(xf_file_url) as f:
        array = []
        for line in f:  # Read rest of lines
            array.append([float(x) for x in line.split()])
        xf = np.array(array)
        # print(xf)
        return xf


class LabelsParser:
    def __init__(self):
        pass

    @staticmethod
    def allocate_labels(img_num):
        return np.empty(img_num, dtype=np.object)

    @staticmethod
    def read_label(cur_file):
        try:
            pose = load_pose_file(cur_file)
            if pose is None:
                return None

            xf = load_xf_file(cur_file)
            if xf is None:
                return None

            label = np.array([pose, xf])
            return label

        except Exception as e:
            print("Unexpected error:", e)
            return None


def show_depth_img(img):
    max = img.max()
    depth_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1,
                              mask=(img != max).astype(np.uint8))

    depth_img[img == max] = 255

    cv2.imshow('depth_image', depth_img)
    cv2.waitKey(0)


class ImageProcessor:
    def __init__(self, x_type):
        self.x_type = x_type

    @staticmethod
    def _add_random_lines(img, lines_per_image=5):
        # import cv2
        import random

        width = img.shape[1]
        height = img.shape[0]

        for n in xrange(lines_per_image):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            cv2.line(img, (x1, y1), (x2, y2), 255)

    @staticmethod
    def _load_face_image(cur_file, image_size, flip=True):
        face_image = load_image(cur_file[:-len('_edges.png')] + '_faces.png', image_size)
        return flip_imgs_colors(face_image) if flip else face_image

    @staticmethod
    def _load_depth_map(cur_file, image_size):
        file_url = cur_file[:-len('_edges.png')] + '_depth.exr'
        depth_map = cv2.imread(file_url, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth_map.shape != image_size:
            # print("Resizing image on load: " + file_url + ", original size  " + str(img.shape))
            depth_map = cv2.resize(depth_map, image_size, interpolation=cv2.INTER_AREA)
        return depth_map

    def allocate_images(self, img_num, image_size):
        if self.x_type == 'depth':
            return np.empty((img_num, image_size[1], image_size[0], 3), dtype=np.float32)
        elif self.x_type == 'stacked_faces':
            return np.empty((img_num, image_size[1], image_size[0], 2), dtype=np.float32)
        else:
            return np.empty((img_num, image_size[1], image_size[0], 1), dtype=np.float32)

    def process(self, img, cur_file, label):
        try:
            img = flip_imgs_colors(img)

            # TODO: Comment or add as option
            # self._add_random_lines(img, lines_per_image=5)

            if self.x_type == 'edges':
                pass

            elif self.x_type == 'gauss_blur_15':
                img = apply_smoothing(img, 15)
                face_image = self._load_face_image(cur_file, utils.get_image_size(img))
                img = cv2.bitwise_and(img, face_image)

            elif self.x_type == 'edges_on_faces':
                face_image = self._load_face_image(cur_file, utils.get_image_size(img), flip=False)
                img = cv2.bitwise_or(img, face_image)
                # imshow('edges_on_faces', img)

            elif self.x_type == 'stacked_faces':
                img_stack = np.empty((img.shape[0], img.shape[1], 2), dtype=np.float32)
                img_stack[:, :, 0] = img
                img_stack[:, :, 1] = self._load_face_image(cur_file, utils.get_image_size(img), flip=False)
                img = img_stack

            elif self.x_type == 'depth':
                img_stack = np.empty((img.shape[0], img.shape[1], 3), dtype=np.float32)
                img_stack[:, :, 0] = img
                img_stack[:, :, 1] = self._load_face_image(cur_file, utils.get_image_size(img), flip=False)
                img_stack[:, :, 2] = self._load_depth_map(cur_file, utils.get_image_size(img))
                # show_depth_img(img_stack[:, :, 2])
                img = img_stack

            else:
                raise Exception("Unknown x_type")

            return img

        except Exception as e:
            print("Unexpected error:", e)
            return None


def process_labels(y_train, y_test, y_type):
    print("Processing labels. y_type [%s]..." % y_type)
    # Get pose - XYZ and angles
    pose_train = np.array([item[0] for item in y_train], dtype=np.float32)
    pose_test = np.array([item[0] for item in y_test], dtype=np.float32)

    # Point to XY columns
    xy_train = pose_train[:, :2]
    xy_test = pose_test[:, :2]

    # Get rotation matrices from projection matrices
    rot_mat_train = np.array([item[1][:3, :3] for item in y_train], dtype=np.float32)
    rot_mat_test = np.array([item[1][:3, :3] for item in y_test], dtype=np.float32)

    if y_type == 'angle':
        # Remove Z and Roll columns from Y
        y_train = np.delete(pose_train, [2, 5], 1)
        y_test = np.delete(pose_test, [2, 5], 1)

    elif y_type == 'quaternion':
        # Important: I modified pyquaternion library sources to be less hard on the rotation matrix orthogonality
        # This is the comment I inserted and the line I edited (Notice the atol=1.e-7):
        # Moti added bigger atol 1.e-7 instead of 1.e-8 to allow conversions of my rotation matrices
        # if not np.allclose(np.dot(R, R.conj().transpose()), np.eye(3), atol=1.e-7):
        quaternion_train = np.array([Quaternion(matrix=item).normalised.elements for item in rot_mat_train],
                                    dtype=np.float32)
        y_train = np.concatenate((xy_train, quaternion_train), axis=1)

        quaternion_test = np.array([Quaternion(matrix=item).normalised.elements for item in rot_mat_test],
                                   dtype=np.float32)
        y_test = np.concatenate((xy_test, quaternion_test), axis=1)

    elif y_type == 'matrix':
        flat_rot_mat_train = np.array([item.flatten() for item in rot_mat_train])
        y_train = np.concatenate((xy_train, flat_rot_mat_train), axis=1)

        flat_rot_mat_test = np.array([item.flatten() for item in rot_mat_test])
        y_test = np.concatenate((xy_test, flat_rot_mat_test), axis=1)
    else:
        raise Exception("Unknown y_type")

    # Find Y min max for matching scaling
    y = np.concatenate((y_train, y_test))
    y_min_max = (np.min(y, axis=0), np.max(y, axis=0))

    print("Done processing labels. Shapes (y_train, y_test): ", (y_train.shape, y_test.shape))
    return y_train, y_test, y_min_max


class DataLoader:
    def __init__(self):
        self.train_dir = None
        self.test_dir = None
        self.image_size = None
        self.part_of_data = None
        self.shuffle = True

        self.x_train = None
        self.x_test = None
        self.labels_train = None
        self.labels_test = None
        self.y_train = None
        self.y_test = None
        self.file_urls_train = None
        self.file_urls_test = None

        self.x_range = None
        self.y_range = None
        self.y_min_max = None

        self.x_type = None
        self.y_type = None

    def load(self, train_dir, test_dir=None, image_size=IMAGE_SIZE, x_range=(0, 1), y_range=(0, 1),
             x_type='edges', y_type='angle', part_of_data=1.0):

        if x_type == 'depth':
            print("Setting x_range=None since x_type=='depth' - Depth is not in image range [0, 255]")
            x_range = None

        print("Load entered. train_dir [%s], test_dir [%s], image_size [%s], x_range [%s], y_range [%s], x_type [%s], "
              "y_type [%s], part_of_data [%s]" % (train_dir, test_dir, image_size, x_range, y_range, x_type, y_type,
                                                  part_of_data))

        self.train_dir = train_dir
        self.test_dir = test_dir

        self.image_size = image_size
        self.x_range = x_range
        self.y_range = y_range

        self.part_of_data = part_of_data

        self.x_type = x_type
        self.y_type = y_type

        # y_type is done after cache is loaded so no need to use in prefix
        save_cache = True
        cache_prefix = "%s" % x_type
        self.x_train, self.labels_train, self.file_urls_train = \
            utils.load_folder(cache_prefix, train_dir, image_size, ext_list='_edges.png', part_of_data=part_of_data,
                              shuffle=True, recursive=True, save_cache=save_cache, skip_upscale=True,
                              labels_parser=LabelsParser, process_image_fn=ImageProcessor(x_type))

        if test_dir is not None:
            self.x_test, self.labels_test, self.file_urls_test = \
                utils.load_folder(cache_prefix, test_dir, image_size, ext_list='_edges.png', part_of_data=part_of_data,
                                  shuffle=True, recursive=True, save_cache=save_cache, skip_upscale=True,
                                  labels_parser=LabelsParser, process_image_fn=ImageProcessor(x_type))
        else:
            print("Test dir not supplied. Splitting train data...")
            self.x_train, _, self.x_test, self.labels_train, _, self.labels_test, self.file_urls_train, _, \
                self.file_urls_test = \
                split.split_data(self.x_train, self.labels_train, 0, 0.2, file_names=self.file_urls_train)

        print("Precessing data...")
        self.y_train, self.y_test, self.y_min_max = process_labels(self.labels_train, self.labels_test, y_type)
        self.x_train, self.y_train, self.file_urls_train = self.process_data(self.x_train, self.y_train,
                                                                             self.file_urls_train)
        self.x_test, self.y_test, self.file_urls_test = self.process_data(self.x_test, self.y_test, self.file_urls_test)

        # import visualize
        # visualize.show_data(x_test, border_size=1, bg_color=(0, 255, 0))

        print("Done DataLoader init. Samples number (train, test): (%s, %s)" %
              (self.x_train.shape[0], self.x_test.shape[0]))

    def process_data(self, x, y, file_urls):
        print("Normalizing data...")
        x, y = self.normalize_data(x, y, self.x_range, self.y_range, self.y_min_max, self.y_type)

        # Not really needed since file list is shuffled before actual load.
        # Notice this is not done in-place, so on very large data-sets the memory struggles...
        # if self.shuffle:
        #     x, y, file_urls = sk_shuffle(x, y, file_urls)

        return x, y, file_urls

    @staticmethod
    def normalize_data(x, y, x_range, y_range, y_min_max, y_type):
        if y.shape[1] not in [4, 6, 11]:  # Sanity
            # Must be 4-(x, y, yaw, pitch), 6-(x, y, quaternion...), 11-(x, y, rotation matrix_3xx3...)
            raise Exception("Invalid y dims [%d] not in [4, 6, 11]" % y.shape[1])

        # (x, y) data
        for i in xrange(2):
            y[:, i] = utils.min_max_scale(y[:, i], (y_min_max[0][i], y_min_max[1][i]), y_range)

        # y[:, 2:] is (Yaw, Pitch)/(Quaternion)/(Rotation matrix flattened)
        if y_type == 'angle':  # Convert negative angles to 0-360, then take to range [0, 1]
            y[:, 2:] = ((y[:, 2:] + 360) % 360) / 360.0

        if x_range is not None:
            x = utils.min_max_scale(x, consts.IMAGE_RANGE, x_range)
        return x, y

    def x_inverse_transform(self, x):
        if self.x_range is None:
            return x
        else:
            return utils.min_max_scale(x, self.x_range, consts.IMAGE_RANGE)

    def y_inverse_transform(self, y):
        if y.ndim == 1:
            y = np.expand_dims(y, axis=0)

        y_new = np.zeros_like(y)

        for i in xrange(2):
            y_new[:, i] = utils.min_max_scale(y[:, i], self.y_range, (self.y_min_max[0][i], self.y_min_max[1][i]))

        if self.y_type == 'angle':
            y_new[:, 2:] = (y[:, 2:] * 360) % 360
        else:
            y_new[:, 2:] = y[:, 2:]
        return y_new.squeeze()


# For testing purpose
def main():
    print("Loading images and labels")

    data_dir = '/home/moti/cg/project/sessions_outputs/berlinRoi_4400_5500_800_800/'
    train_dir = os.path.join(data_dir, 'train')
    test_dir = None
    # test_dir = os.path.join(data_dir, 'test')

    loader = DataLoader()
    loader.load(train_dir=train_dir,
                test_dir=test_dir,
                x_range=(0, 1),
                x_type='edges',
                y_type='angle',
                part_of_data=100)

    # visualize.show_data(loader.x_train, bg_color=(0, 255, 0))

    # visualize.show_data(x_train, border_size=1, bg_color=(0, 255, 0))

    # for i in range(0, x.shape[0]):
    #     title = str(i) + " - " + file_urls[i]
    #     cv2.imshow(title, x[i])
    #     key = cv2.waitKey(0) & 0xFF
    #     cv2.destroyWindow(title)
    #     if key == 27:
    #         break

    print("bye")


if __name__ == '__main__':
    main()
