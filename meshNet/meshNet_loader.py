from __future__ import print_function
from __future__ import division

import os
import random
import numpy as np
import cv2
from pyquaternion import Quaternion
from tqdm import tqdm

import utils
import split
import consts
# from visualize import imshow

IMAGE_SIZE = (int(800/5), int(600/5))
# IMAGE_SIZE = (int(800/5), int((600/3)/5))

EDGE_EXT = '_edges.png'
FACE_EXT = '_faces.png'
DEPTH_EXT = '_depth.exr'


class LabelsParser:
    def __init__(self):
        pass

    @staticmethod
    def allocate_labels(img_num):
        return np.empty(img_num, dtype=np.object)

    @staticmethod
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

    @staticmethod
    def load_xf_file(file_url):
        xf_file_url = os.path.splitext(file_url)[0][:-len('_edges')] + '.xf'
        with open(xf_file_url) as f:
            array = []
            for line in f:  # Read rest of lines
                array.append([float(x) for x in line.split()])
            xf = np.array(array)
            # print(xf)
            return xf

    @staticmethod
    def read_label(cur_file):
        try:
            pose = LabelsParser.load_pose_file(cur_file)
            if pose is None:
                return None

            xf = LabelsParser.load_xf_file(cur_file)
            if xf is None:
                return None

            label = np.array([pose, xf])
            return label

        except Exception as e:
            print("Unexpected error:", e)
            return None


class ImageProcessor:
    def __init__(self):
        pass

    @staticmethod
    def allocate_images(img_num, image_size):
        x_edges = np.empty((img_num, image_size[1], image_size[0], 1), dtype=np.uint8)
        x_faces = np.empty((img_num, image_size[1], image_size[0], 1), dtype=np.uint8)
        x_depth = np.empty((img_num, image_size[1], image_size[0], 1), dtype=np.float32)
        return x_edges, x_faces, x_depth

    @staticmethod
    def load_image(file_url, image_size, skip_upscale=True):
        img = cv2.imread(file_url, cv2.IMREAD_GRAYSCALE)

        if img.shape[1] / float(img.shape[0]) != image_size[0] / float(image_size[1]):
            print("Wrong aspect ratio. Expected %f, received %f" % (image_size[1] / float(image_size[0]),
                                                                    img.shape[1] / float(img.shape[0])))
            return None

        if skip_upscale and (img.shape[0] < image_size[0] or img.shape[1] < image_size[1]):
            print("Skipping small image. Expected (%s,%s), received (%s,%s)" %
                  (image_size[0], image_size[1], img.shape[0], img.shape[1]))
            return None

        if img.shape != image_size:
            # print("Resizing image on load: " + file_url + ", original size  " + str(img.shape))
            img = cv2.resize(img, image_size, interpolation=cv2.INTER_AREA)

        if len(img.shape) == 2:
            img = np.expand_dims(img, axis=-1)

        return img

    @staticmethod
    def load_face_image(file_url, image_size, flip):
        face_image = ImageProcessor.load_image(file_url[:-len(EDGE_EXT)] + FACE_EXT, image_size)
        return ImageProcessor.flip_imgs_colors(face_image) if flip else face_image

    @staticmethod
    def load_depth_map(file_url, image_size):
        depth_map = cv2.imread(file_url[:-len(EDGE_EXT)] + DEPTH_EXT, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
        if depth_map.shape != image_size:
            # print("Resizing image on load: " + file_url + ", original size  " + str(img.shape))
            depth_map = cv2.resize(depth_map, image_size, interpolation=cv2.INTER_AREA)

        if len(depth_map.shape) == 2:
            depth_map = np.expand_dims(depth_map, axis=-1)

        return depth_map

    @staticmethod
    def show_depth_img(img, wait_key=True):
        max_val = img.max()
        depth_img = cv2.normalize(img, None, alpha=0, beta=255, norm_type=cv2.NORM_MINMAX, dtype=cv2.CV_8UC1,
                                  mask=(img != max_val).astype(np.uint8))

        depth_img[img == max_val] = 255

        cv2.imshow('depth', depth_img)
        if wait_key:
            cv2.waitKey(0)

    @staticmethod
    def show_images(x):
        print('Entered show_images(): Press Esc to quit')

        for i in xrange(len(x)):
            if x.shape[3] == 1:
                cv2.imshow('edge', x[i, :, :, 0].astype(np.uint8))

            if x.shape[3] == 2:
                cv2.imshow('edge', x[i, :, :, 0].astype(np.uint8))
                cv2.imshow('face', x[i, :, :, 1].astype(np.uint8))

            if x.shape[3] == 3:
                cv2.imshow('edge', x[i, :, :, 0].astype(np.uint8))
                cv2.imshow('face', x[i, :, :, 1].astype(np.uint8))
                ImageProcessor.show_depth_img(x[i, :, :, 2], wait_key=False)

            key = cv2.waitKey(0)
            if key == 27:  # Esc
                break

    @staticmethod
    def apply_smoothing(image, kernel_size=7):
        """
        kernel_size must be positive and odd
        """
        return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)

    @staticmethod
    def flip_imgs_colors(imgs):
        # Flip image colors (White <-> Black)
        white_imgs = np.full_like(imgs, 255)
        imgs = white_imgs - imgs
        imgs[imgs > 0] = 255  # Make sure all values are 0/255
        return imgs

    @staticmethod
    def add_random_lines(img, lines_per_image=5):
        width = img.shape[1]
        height = img.shape[0]

        for n in xrange(lines_per_image):
            x1 = random.randint(0, width)
            y1 = random.randint(0, height)
            x2 = random.randint(0, width)
            y2 = random.randint(0, height)
            cv2.line(img, (x1, y1), (x2, y2), 255)

    @staticmethod
    def process(x_edges, x_faces, x_depth, x_type):

        # TODO: Comment or add as option
        # self._add_random_lines(img, lines_per_image=5)

        if x_type == 'edges':
            x = ImageProcessor.flip_imgs_colors(x_edges)
            x = x.astype(np.float32)

        elif x_type == 'faces':
            x_faces = ImageProcessor.flip_imgs_colors(x_faces)
            x = x_faces.astype(np.float32)

        elif x_type == 'gauss_blur_15':
            x = np.empty((len(x_edges), x_edges.shape[1], x_edges.shape[2], 1), dtype=np.float32)

            for i in xrange(len(x_edges)):
                img = x_edges[i]
                img = ImageProcessor.flip_imgs_colors(img)
                img = ImageProcessor.apply_smoothing(img, 15)

                face_image = x_faces[i]
                face_image = ImageProcessor.flip_imgs_colors(face_image)

                img = cv2.bitwise_and(img, face_image)
                x[i, :, :, 0] = img

        elif x_type == 'edges_on_faces':
            x = np.empty((len(x_edges), x_edges.shape[1], x_edges.shape[2], 1), dtype=np.float32)

            for i in xrange(len(x_edges)):
                img = x_edges[i]
                img = ImageProcessor.flip_imgs_colors(img)

                face_image = x_faces[i]
                img = cv2.bitwise_or(img, face_image)
                # imshow('edges_on_faces', img)
                x[i, :, :, 0] = img

        elif x_type == 'stacked_faces':
            x = np.empty((len(x_edges), x_edges.shape[1], x_edges.shape[2], 2), dtype=np.float32)
            x[:, :, :, 0] = x_edges[:, :, :, 0]
            x[:, :, :, 1] = x_faces[:, :, :, 0]

        elif x_type == 'depth':
            x = np.empty((len(x_edges), x_edges.shape[1], x_edges.shape[2], 3), dtype=np.float32)
            x[:, :, :, 0] = x_edges[:, :, :, 0]
            x[:, :, :, 1] = x_faces[:, :, :, 0]
            x[:, :, :, 2] = x_depth[:, :, :, 0]
            # ImageProcessor.show_depth_img(x[0, :, :, 2])

        else:
            raise Exception("Unknown x_type")

        # ImageProcessor.show_images(x)
        return x


def load_file_list(files, data_dir, image_size):
    img_num = len(files)
    print("Loading file list of size", img_num)

    x_edges = None
    x_faces = None
    x_depth = None

    labels = LabelsParser.allocate_labels(img_num)
    file_urls = np.empty(img_num, dtype="object")
    if image_size is not None:
        x_edges, x_faces, x_depth = ImageProcessor.allocate_images(img_num, image_size)

    cnt = 0
    for cur_file in tqdm(files):
        if image_size is not None:
            edge_img = ImageProcessor.load_image(cur_file, image_size)
            face_img = ImageProcessor.load_face_image(cur_file, image_size, flip=False)
            depth_img = ImageProcessor.load_depth_map(cur_file, image_size)
            if edge_img is None or face_img is None or depth_img is None:
                print("One of the images are invalid. Skipping")
                continue

            x_edges[cnt, :, :, :] = edge_img
            x_faces[cnt, :, :, :] = face_img
            x_depth[cnt, :, :, :] = depth_img

        label = LabelsParser.read_label(cur_file)
        if label is None:
            print(cur_file + " label is invalid. Skipping")
            continue

        labels[cnt] = label

        file_urls[cnt] = os.path.join(data_dir, cur_file)

        cnt += 1

    if image_size is not None:
        x_edges = x_edges[:cnt]
        x_faces = x_faces[:cnt]
        x_depth = x_depth[:cnt]

    labels = labels[:cnt]
    file_urls = file_urls[:cnt]

    print("Loaded " + str(len(file_urls)) + " images from " + data_dir)
    return (labels, file_urls) if image_size is None else (x_edges, x_faces, x_depth, labels, file_urls)


def load_data(data_dir, image_size, x_type, part_of_data=1.0, recursive=True, use_cache=True):
    print("Loading data_dir [%s], image_size [%s], x_type [%s], part_of_data [%s], recursive [%s], use_cache [%s]"
          % (data_dir, image_size, x_type, part_of_data, recursive, use_cache))

    if not os.path.isdir(data_dir):
        raise Exception("Directory " + data_dir + " does not exist")

    with utils.cd(data_dir):
        params_prefix = str(part_of_data)
        params_prefix += "_%sx%s" % (image_size[0], image_size[1]) if image_size is not None else ""
        params_prefix += "_r" if recursive else ""
        print("Cache params_prefix:", params_prefix)

        x_edges = None
        x_faces = None
        x_depth = None
        if use_cache and (image_size is not None and
                          (os.path.isfile('edges_' + params_prefix + "_x.npy") and
                           os.path.isfile('faces_' + params_prefix + "_x.npy") and
                           os.path.isfile('depth_' + params_prefix + "_x.npy"))
                          or image_size is None and os.path.isfile(params_prefix + "_file_urls.npy")):
            print("Loading cache...")
            if image_size is not None:
                x_edges = np.load('edges_' + params_prefix + "_x.npy")
                x_faces = np.load('faces_' + params_prefix + "_x.npy")
                x_depth = np.load('depth_' + params_prefix + "_x.npy")

            file_urls = np.load(params_prefix + "_file_urls.npy")
            labels = np.load(params_prefix + "_labels.npy")
            print("Done loading cache")
        else:
            print("Loading files...")
            files = utils.get_files_with_ext(data_dir, ext_list=EDGE_EXT, recursive=recursive, sort=False)

            files = utils.part_of(files, part=part_of_data)

            if image_size is None:
                labels, file_urls = load_file_list(files, data_dir, image_size)
            else:
                x_edges, x_faces, x_depth, labels, file_urls = load_file_list(files, data_dir, image_size)

            if use_cache:
                try:
                    print("Trying to save cache...")
                    if image_size is not None:
                        np.save('edges_' + params_prefix + "_x.npy", x_edges)
                        np.save('faces_' + params_prefix + "_x.npy", x_faces)
                        np.save('depth_' + params_prefix + "_x.npy", x_depth)

                    np.save(params_prefix + "_labels.npy", labels)

                    np.save(params_prefix + "_file_urls.npy", file_urls)
                    print("Cache saved")
                except Exception as e:
                    print(e)
                    print("Cache NOT saved due to exception. Probably no space left on disk")

    x = None
    if image_size is not None:
        x = ImageProcessor.process(x_edges, x_faces, x_depth, x_type)

    print("Loaded " + str(len(file_urls)) + " images from " + data_dir)
    return (labels, file_urls) if image_size is None else (x, labels, file_urls)


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


def auto_collect_data(data_dir, image_size, x_type, roi, grid_step, part_of_data=1.0, use_cache=True):
    print("Auto collecting data_dir [%s], image_size [%s], x_type [%s], roi [%s], grid_step [%s], part_of_data [%s]"
          % (data_dir, image_size, x_type, roi, grid_step, part_of_data))

    if not os.path.isdir(data_dir):
        raise Exception("Directory " + data_dir + " does not exist")

    x_edges_train = x_faces_train = x_depth_train = x_edges_test = x_faces_test = x_depth_test = None
    with utils.cd(data_dir):
        params_prefix = str(part_of_data)
        params_prefix += "_%sx%s" % (image_size[0], image_size[1]) if image_size is not None else ""
        params_prefix += "_%s_%s_%s_%s_step_%s" % (roi[0], roi[1], roi[2], roi[3], grid_step)
        print("Cache params_prefix:", params_prefix)

        if use_cache and (image_size is not None and
                          (os.path.isfile(params_prefix + "_x_edges_train.npy") and
                           os.path.isfile(params_prefix + "_x_faces_train.npy") and
                           os.path.isfile(params_prefix + "_x_depth_train.npy") and
                           os.path.isfile(params_prefix + "_x_edges_test.npy") and
                           os.path.isfile(params_prefix + "_x_faces_test.npy") and
                           os.path.isfile(params_prefix + "_x_depth_test.npy"))
                          or image_size is None and os.path.isfile(params_prefix + "_file_urls_train.npy") and
                          os.path.isfile(params_prefix + "_file_urls_test.npy")):
            print("Loading cache...")
            if image_size is not None:
                x_edges_train = np.load(params_prefix + "_x_edges_train.npy")
                x_faces_train = np.load(params_prefix + "_x_faces_train.npy")
                x_depth_train = np.load(params_prefix + "_x_depth_train.npy")
                x_edges_test = np.load(params_prefix + "_x_edges_test.npy")
                x_faces_test = np.load(params_prefix + "_x_faces_test.npy")
                x_depth_test = np.load(params_prefix + "_x_depth_test.npy")

            file_urls_train = np.load(params_prefix + "_file_urls_train.npy")
            file_urls_test = np.load(params_prefix + "_file_urls_test.npy")

            labels_train = np.load(params_prefix + "_labels_train.npy")
            labels_test = np.load(params_prefix + "_labels_test.npy")
            print("Done loading cache")
        else:
            file_urls_cache_fname = "auto_collect_file_urls.npy"
            if os.path.isfile(file_urls_cache_fname):
                print("Loading file_urls from cache")
                file_urls = np.load(file_urls_cache_fname)
            else:
                print("Loading file_urls (This might take a while)...")
                file_urls = utils.get_files_with_ext(data_dir, ext_list=EDGE_EXT, recursive=True, sort=False)

                print("Saving file_urls cache")
                np.save(file_urls_cache_fname, file_urls)

            print("Total edge images number:", len(file_urls))
            file_urls = utils.part_of(file_urls, part=part_of_data)
            print("Part of data edge images number:", len(file_urls))

            file_urls_train = np.empty(len(file_urls), dtype="object")
            file_urls_test = np.empty(len(file_urls), dtype="object")
            cnt_train = 0
            cnt_test = 0
            for f in file_urls:
                # Get current image (x, y) position
                x, y = os.path.split(os.path.split(f)[0])[1].split('_')
                x = int(x)
                y = int(y)

                # Check if position should be included in either train/test sets
                if x < roi[0] or x > roi[0]+roi[2] or y < roi[1] or y > roi[1]+roi[3]:
                    continue

                x -= roi[0]
                y -= roi[1]
                if x % grid_step == 0 and y % grid_step == 0:
                    file_urls_train[cnt_train] = f
                    cnt_train += 1
                elif (x-grid_step/2) % grid_step == 0 and (y-grid_step/2) % grid_step == 0:
                    file_urls_test[cnt_test] = f
                    cnt_test += 1
                else:
                    continue

            file_urls_train = file_urls_train[:cnt_train]
            file_urls_test = file_urls_test[:cnt_test]

            if image_size is None:
                labels_train, file_urls_train = load_file_list(file_urls_train, data_dir, image_size)
                labels_test, file_urls_test = load_file_list(file_urls_test, data_dir, image_size)
            else:
                x_edges_train, x_faces_train, x_depth_train, labels_train, file_urls_train = \
                    load_file_list(file_urls_train, data_dir, image_size)
                x_edges_test, x_faces_test, x_depth_test, labels_test, file_urls_test = \
                    load_file_list(file_urls_test, data_dir, image_size)

            if use_cache:
                try:
                    print("Trying to save cache...")
                    if image_size is not None:
                        np.save(params_prefix + "_x_edges_train.npy", x_edges_train)
                        np.save(params_prefix + "_x_faces_train.npy", x_faces_train)
                        np.save(params_prefix + "_x_depth_train.npy", x_depth_train)
                        np.save(params_prefix + "_x_edges_test.npy", x_edges_test)
                        np.save(params_prefix + "_x_faces_test.npy", x_faces_test)
                        np.save(params_prefix + "_x_depth_test.npy", x_depth_test)

                    np.save(params_prefix + "_labels_train.npy", labels_train)
                    np.save(params_prefix + "_labels_test.npy", labels_test)

                    np.save(params_prefix + "_file_urls_train.npy", file_urls_train)
                    np.save(params_prefix + "_file_urls_test.npy", file_urls_test)
                    print("Cache saved")
                except Exception as e:
                    print(e)
                    print("Cache NOT saved due to exception. Probably no space left on disk")

    x_train = x_test = None
    if image_size is not None:
        x_train = ImageProcessor.process(x_edges_train, x_faces_train, x_depth_train, x_type)
        x_test = ImageProcessor.process(x_edges_test, x_faces_test, x_depth_test, x_type)

    print("Loaded " + str(len(file_urls_train) + len(file_urls_test)) + " images from " + data_dir)
    return (labels_train, file_urls_train, labels_test, file_urls_test) if image_size is None else \
        (x_train, labels_train, file_urls_train, x_test, labels_test, file_urls_test)

# Keep only part of the samples. Makes a sparser 4D-grid
def filter_angles(x_train, labels_train, file_urls_train):
    yaw_list = np.linspace(0, 360, 9)
    pitch_list = [-6, -12]
    print("Filtering angles. Leaving pitch", pitch_list, "and yaw", yaw_list)

    x_train_new = []
    labels_train_new = []
    file_urls_train_new = []

    for i, label in enumerate(labels_train):
        if label[0][3] in yaw_list and label[0][4] in pitch_list:
            x_train_new.append(x_train[i])
            labels_train_new.append(label)
            file_urls_train_new.append(file_urls_train[i])

    x_train_new = np.asarray(x_train_new)
    labels_train_new = np.asarray(labels_train_new)
    file_urls_train_new = np.asarray(file_urls_train_new)

    return x_train_new, labels_train_new, file_urls_train_new


# Keep some percentage of the test samples. Used for faster training
def filter_tests(x_test, labels_test, file_urls_test, p=0.3):
    print("Filtering test set. p=", p)

    x_test_new = []
    labels_test_new = []
    file_urls_test_new = []
    for i, label in enumerate(labels_test):
        c = np.random.rand(1)
        if c < p:
            x_test_new.append(x_test[i])
            labels_test_new.append(label)
            file_urls_test_new.append(file_urls_test[i])

    x_test_new = np.asarray(x_test_new)
    labels_test_new = np.asarray(labels_test_new)
    file_urls_test_new = np.asarray(file_urls_test_new)

    return x_test_new, labels_test_new, file_urls_test_new


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
             x_type='edges', y_type='angle', part_of_data=1.0, use_cache=True, auto_collect=True, roi=None,
             grid_step=None):

        if x_type == 'depth':
            print("Setting x_range=None since x_type=='depth' - Depth is not in image range [0, 255]")
            x_range = None

        if auto_collect and (roi is None or grid_step is None):
            raise Exception("On auto_collect roi and grid_step must be provided")

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

        if auto_collect:
            data_dir = os.path.split(train_dir)[0]
            self.x_train, self.labels_train, self.file_urls_train, \
                self.x_test, self.labels_test, self.file_urls_test = auto_collect_data(data_dir, image_size, x_type,
                                                                                       roi, grid_step,
                                                                                       part_of_data=part_of_data)
        else:
            self.x_train, self.labels_train, self.file_urls_train = \
                load_data(train_dir, image_size, x_type, part_of_data=part_of_data, use_cache=use_cache)

            if test_dir is not None:
                self.x_test, self.labels_test, self.file_urls_test = \
                    load_data(test_dir, image_size, x_type, part_of_data=part_of_data, use_cache=use_cache)
            else:
                print("Test dir not supplied. Splitting train data...")
                self.x_train, _, self.x_test, self.labels_train, _, self.labels_test, self.file_urls_train, _, \
                    self.file_urls_test = \
                    split.split_data(self.x_train, self.labels_train, 0, 0.2, file_names=self.file_urls_train)

        # self.x_train, self.labels_train, self.file_urls_train = filter_angles(
        #     self.x_train, self.labels_train, self.file_urls_train)

        # self.x_test, self.labels_test, self.file_urls_test = filter_tests(
        #     self.x_test, self.labels_test, self.file_urls_test)

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
