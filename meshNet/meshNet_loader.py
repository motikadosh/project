from __future__ import print_function

import os
import pickle
import numpy as np
import cv2

import utils
import split
import visualize

from sklearn.preprocessing import MinMaxScaler


IMAGE_SIZE = (100, 75)
IMAGE_MINIAMAL_CONTENT = 0.1  # Ignore images with less than this percent of pixels


def save_data_pkl(file_path, x_train, x_test,  y_train, y_test, file_urls_train, file_urls_test):
    if not file_path:  # Return if None or empty
        return

    if os.path.exists(file_path):
        print("File already exists [" + file_path + "]. Aborting...")
        return

    print("Saving data to pickle [" + file_path + "]")

    data = [x_train, x_test,  y_train, y_test, file_urls_train, file_urls_test]
    with open(file_path, "wb") as f:
        pickle.dump(data, f)


def load_data_pkl(file_path):
    print("Loading data from pickle [" + file_path + "]")

    with open(file_path) as f:
        data = pickle.load(f)
        x_train = data[0]
        x_test = data[1]
        y_train = data[2]
        y_test = data[3]
        file_urls_train = data[4]
        file_urls_test = data[5]

        return x_train, x_test, y_train, y_test, file_urls_train, file_urls_test


def load_xf_file(file_url):
    xf_file_url = os.path.splitext(file_url)[0] + '.xf'
    with open(xf_file_url) as f:
        array = []
        for line in f:  # read rest of lines
            array.append([float(x) for x in line.split()])
        xf = np.array(array)
        # print(xf)
        return xf.reshape(-1)


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


def resize_batch(x, new_size):
    img_num = x.shape[0]
    new_x = np.empty((img_num, new_size[1], new_size[0], 1), dtype='uint8')

    for i in range(img_num):
        new_img = cv2.resize(x[i], new_size, interpolation=cv2.INTER_AREA)
        if new_img.ndim == 2:  # Handle adding the channels axis to a single channel image
            new_img = np.expand_dims(new_img, 2)
        new_x[i] = new_img

    return new_x


def allocate_res_arrays(img_num):
    images = np.empty((img_num, IMAGE_SIZE[1], IMAGE_SIZE[0], 1), dtype='uint8')
    # xf_matrices = np.empty((img_num, 16), dtype=np.float32)
    poses = np.empty((img_num, 6), dtype=np.float32)
    file_urls = np.empty(img_num, dtype="object")
    return images, poses, file_urls


def load_data(data_dir, part_of_data=1.0, shuffle=True, pkl_file_path=None, image_size=IMAGE_SIZE):
    print("Loading data...")
    if part_of_data > 1 or part_of_data <= 0:
        raise Exception("Invalid argument, should be in (0,1]")

    if pkl_file_path is not None and os.path.exists(pkl_file_path):
        x_train, x_test, y_train, y_test, file_urls_train, file_urls_test = load_data_pkl(pkl_file_path)

        x_train = x_train[:int(len(x_train) * part_of_data)]
        x_test = x_test[:int(len(x_test) * part_of_data)]

        if image_size != IMAGE_SIZE:
            x_train = resize_batch(x_train, image_size)
            x_test = resize_batch(x_test, image_size)
        return x_train, x_test, y_train, y_test, file_urls_train, file_urls_test

    image_files = utils.get_files_with_ext(data_dir)
    print("Found " + str(len(image_files)) + " images in " + data_dir)

    if shuffle:
        np.random.shuffle(image_files)  # Prevent following code getting images from certain type
    if part_of_data < 1:
        image_files = image_files[:int(len(image_files) * part_of_data)]

    img_num = len(image_files)
    x, y, file_urls = allocate_res_arrays(img_num)

    cnt = 0
    for file_url in image_files:
        x[cnt, :, :, 0] = load_image(file_url)
        # y[cnt] = load_xf_file(file_url)
        y[cnt] = load_pose_file(file_url)
        file_urls[cnt] = file_url

        if cnt % 100 == 0:
            print("Loaded " + str(cnt) + "/" + str(len(image_files)) + " images")

        # Ignore images with less than IMAGE_MINIAMAL_CONTENT percent of pixels
        number_of_pixels_with_data = (x[cnt] != 0).sum()
        data_percent_from_image = number_of_pixels_with_data / IMAGE_SIZE[0] * IMAGE_SIZE[1]
        if data_percent_from_image >= IMAGE_MINIAMAL_CONTENT:
            cnt += 1

    # visualize.show_data(x)

    # Clear empty trailing cells
    x = x[:cnt]
    y = y[:cnt]
    file_urls = file_urls[:cnt]

    # Special variables handling
    x = x.astype('float32')
    x = x / 255 - 0.5  # Take x to range [-0.5, 0.5]
    y = np.delete(y, [2, 5], 1)  # Remove Z and Roll columns from Y
    min_max_scaler = MinMaxScaler()
    y = min_max_scaler.fit_transform(y)  # Take y to range [0, 1.0]

    print("Splitting data...")
    x_train, _, x_test, y_train, _, y_test, file_urls_train, _, file_urls_test = split.split_data(
        x, y, 0, 0.2, file_names=file_urls)

    save_data_pkl(pkl_file_path, x_train, x_test, y_train, y_test, file_urls_train, file_urls_test)
    return x_train, x_test, y_train, y_test, file_urls_train, file_urls_test


# For testing purpose
def main():
    print("Loading images and lables")
    x_train, x_test, \
        y_train, y_test, \
        file_urls_train, file_urls_test = load_data(
            '/home/moti/cg/project/sample_images/', part_of_data=0.001, shuffle=False)

    visualize.show_data(x_train)

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

