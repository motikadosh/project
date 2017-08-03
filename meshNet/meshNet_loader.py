from __future__ import print_function
from __future__ import division

import os
import pickle
import numpy as np
import cv2

import utils
import split
import visualize

from sklearn.preprocessing import MinMaxScaler

# IMAGE_SIZE = (int(800/8), int(600/8))
IMAGE_SIZE = (int(800/5), int((600/3)/5))
IMAGE_RANGE = (0, 255)


class DataLoader:
    _IMAGE_MINIAMAL_CONTENT = 0.0  # Ignore images with less than this percent of pixels. 0 - skip this check

    data_dir = None
    image_size = None
    part_of_data = None
    pkl_cache_file_path = None
    shuffle = True

    x_train = None
    x_test = None
    y_train = None
    y_test = None
    file_urls_train = None
    file_urls_test = None

    x_range = None
    y_range = None
    y_min_max_scaler = None

    def __init__(self, data_dir, image_size=IMAGE_SIZE, x_range=(-0.5, 0.5), y_range=(0, 1), part_of_data=1.0,
                 pkl_cache_file_path=None):
        if part_of_data > 1 or part_of_data <= 0:
            raise Exception("Invalid argument, should be in (0,1]")

        self.data_dir = data_dir
        self.image_size = image_size
        self.x_range = x_range
        self.y_range = y_range

        self.part_of_data = part_of_data
        self.pkl_cache_file_path = pkl_cache_file_path

        # Try loading the cache
        # If it is not available load the data with default params
        if not self.load_cache():
            self.load_data()

        # print("Done DataLoader init. Samples number: %s" % (self.x_train.shape[0] + self.x_test.shape[0]))

    def save_cache(self):
        if not self.pkl_cache_file_path:  # Return if None or empty
            return

        if os.path.exists(self.pkl_cache_file_path):
            print("File already exists [" + self.pkl_cache_file_path + "]. Aborting...")
            return

        print("Saving data to pickle [" + self.pkl_cache_file_path + "]...")
        with open(self.pkl_cache_file_path, "wb") as f:
            pickle.dump(self.__dict__, f)
        print("Done saving data to pickle [" + self.pkl_cache_file_path + "]")

    def load_cache(self):
        if not self.pkl_cache_file_path or not os.path.exists(self.pkl_cache_file_path):
            return False

        print("Loading data from pickle [" + self.pkl_cache_file_path + "]...")
        with open(self.pkl_cache_file_path) as f:
            tmp_dict = pickle.load(f)

        self.__dict__.update(tmp_dict)
        print("Done loading data from pickle [" + self.pkl_cache_file_path + "]")
        return True

    def load_data(self):
        print("Loading data...")

        # Get full images list
        image_files = utils.get_files_with_ext(self.data_dir)
        print("Found " + str(len(image_files)) + " images in " + self.data_dir)

        if self.shuffle:
            np.random.shuffle(image_files)  # Prevent following code getting images from certain type
        if self.part_of_data < 1.0:
            image_files = image_files[:int(len(image_files) * self.part_of_data)]

        # Memory allocations
        x, y, file_urls = self.allocate_res_arrays(len(image_files))

        # Data loading
        cnt = 0
        for file_url in image_files:
            x[cnt, :, :, 0] = self.load_image(file_url)
            # y[cnt] = load_xf_file(file_url)
            y[cnt] = self.load_pose_file(file_url)
            file_urls[cnt] = file_url

            if cnt % 100 == 0:
                print("Loaded " + str(cnt) + "/" + str(len(image_files)) + " images")

            if self._IMAGE_MINIAMAL_CONTENT > 0.0:
                # Ignore images with less than IMAGE_MINIAMAL_CONTENT percent of pixels
                number_of_pixels_with_data = (x[cnt] != 0).sum()
                data_percent_from_image = number_of_pixels_with_data / IMAGE_SIZE[0] * IMAGE_SIZE[1]
                if data_percent_from_image >= self._IMAGE_MINIAMAL_CONTENT:
                    cnt += 1
            else:
                cnt += 1

                # visualize.show_data(x)

        # Clear empty trailing cells
        x = x[:cnt]
        y = y[:cnt]
        file_urls = file_urls[:cnt]

        # Special variables handling
        y = np.delete(y, [2, 5], 1)  # Remove Z and Roll columns from Y
        print("Normalizing data...")
        x, y, self.y_min_max_scaler = self.normalize_data(x, y, self.x_range, self.y_range)

        # Splitting data to train/test
        print("Splitting data...")
        self.x_train, _, self.x_test, self.y_train, _, self.y_test, self.file_urls_train, _, self.file_urls_test = \
            split.split_data(x, y, 0, 0.2, file_names=file_urls)

        self.save_cache()
        print("Done loading data")

    def allocate_res_arrays(self, img_num):
        images = np.empty((img_num, self.image_size[1], self.image_size[0], 1), dtype='uint8')
        # xf_matrices = np.empty((img_num, 16), dtype=np.float32)
        poses = np.empty((img_num, 6), dtype=np.float32)
        file_urls = np.empty(img_num, dtype="object")
        return images, poses, file_urls

    def normalize_data(self, x, y, x_range, y_range):
        y_min_max_scaler = MinMaxScaler(feature_range=y_range)
        y = y_min_max_scaler.fit_transform(y)  # Transform y to range in self.y_range

        x = x.astype(np.float32)
        x = self.x_min_max_scale(x, IMAGE_RANGE, x_range)
        return x, y, y_min_max_scaler

    @staticmethod
    def x_min_max_scale(x, old_range, new_range):
        old_span = old_range[1] - old_range[0]
        new_span = new_range[1] - new_range[0]

        if old_span == 0 or new_span == 0:
            raise Exception("Invalid old/new range")

        new_x = (((x - old_range[0]) * new_span) / old_span) + new_range[0]
        return new_x

    def x_inverse_transform(self, x):
        return self.x_min_max_scale(x.astype(np.float32), self.x_range, IMAGE_RANGE)

    def y_inverse_transform(self, y):
        if y.ndim == 1:
            return self.y_min_max_scaler.inverse_transform(y.reshape(1, -1))
        else:
            return self.y_min_max_scaler.inverse_transform(y)

    @staticmethod
    def load_xf_file(file_url):
        xf_file_url = os.path.splitext(file_url)[0] + '.xf'
        with open(xf_file_url) as f:
            array = []
            for line in f:  # read rest of lines
                array.append([float(x) for x in line.split()])
            xf = np.array(array)
            # print(xf)
            return xf.reshape(-1)

    @staticmethod
    def load_pose_file(file_url):
        pose_file_url = os.path.splitext(file_url)[0] + '.txt'
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
    def load_image(file_url):
        img = cv2.imread(file_url, cv2.IMREAD_GRAYSCALE)
        if img.shape != IMAGE_SIZE:
            # print("Resizing image on load: " + file_url + ", original size  " + str(img.shape))
            img = cv2.resize(img, IMAGE_SIZE, interpolation=cv2.INTER_AREA)

            # Flip images colors (White - Black)
            white_img = np.full(img.shape, 255, img.dtype)
            img = white_img - img

            # Make sure all values are 0/255
            img[img > 0] = 255

        return img

    @staticmethod
    def resize_batch(x, new_size):
        img_num = x.shape[0]
        new_x = np.empty((img_num, new_size[1], new_size[0], 1), dtype='uint8')

        for i in range(img_num):
            new_img = cv2.resize(x[i], new_size, interpolation=cv2.INTER_AREA)
            if new_img.ndim == 2:  # Handle adding the channels axis to a single channel image
                new_img = np.expand_dims(new_img, 2)
            new_x[i] = new_img

        return new_x

    def set_part_of_data(self, part_of_data=1.0):
        if part_of_data > 1 or part_of_data <= 0:
            raise Exception("Invalid argument, should be in (0,1]")

        if part_of_data < 1:
            self.x_train = self.x_train[:int(len(self.x_train) * part_of_data)]
            self.y_train = self.y_train[:int(len(self.y_train) * part_of_data)]
            self.file_urls_train = self.file_urls_train[:int(len(self.file_urls_train) * part_of_data)]

            self.x_test = self.x_test[:int(len(self.x_test) * part_of_data)]
            self.y_test = self.y_test[:int(len(self.y_test) * part_of_data)]
            self.file_urls_test = self.file_urls_test[:int(len(self.file_urls_test) * part_of_data)]

    def set_image_size(self, image_size):
        if image_size[0] > self.image_size[0] or image_size[1] > self.image_size[1]:
            raise Exception(
                "Setting bigger image than loaded is prohibited. Remove cache and reload data in larger size")

        if image_size != self.image_size:
            self.x_train = self.resize_batch(self.x_train, image_size)
            self.x_test = self.resize_batch(self.x_test, image_size)


# For testing purpose
def main():
    # print("Loading images and lables")
    # x_train, x_test, \
    #     y_train, y_test, \
    #     file_urls_train, file_urls_test, y_min_max_scaler = load_data(
    #         '/home/moti/cg/project/sample_images/', part_of_data=0.1, shuffle=False, image_size=(100, 75))

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
