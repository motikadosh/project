from __future__ import print_function
from __future__ import division

import time
import random

import numpy as np
from scipy.misc import comb
import matplotlib.pyplot as plt
import cv2

import meshNet_loader
import visualize


data_dir = '/home/moti/cg/project/sessions_outputs/project_2017_08_08-13_11_31-500samplesMin5LinesWith50PixUpper0.33'


def calc_ssd(img1, img2):
    ssd = np.sum((img1 - img2) ** 2)
    return ssd


def apply_smoothing(image, kernel_size=7):
    """
    kernel_size must be positive and odd
    """
    return cv2.GaussianBlur(image, (kernel_size, kernel_size), 0)


# http://www.pyimagesearch.com/2016/11/07/intersection-over-union-iou-for-object-detection/
def calc_iou(img1, img2, smooth_kernel=None):
    if smooth_kernel is not None:
        img1 = apply_smoothing(img1, kernel_size=smooth_kernel)
        img2 = apply_smoothing(img2, kernel_size=smooth_kernel)
    # visualize.show_data(np.array([img1, img2]))

    intersection = np.sum(np.logical_and(img1, img2))
    union = np.sum(np.logical_or(img1, img2))
    iou = intersection / union
    return iou


def plot_hist(x, normed, bins, title='Histogram', ylabel=None):
    print ("Plot: " + title)

    if ylabel is None and normed:
        ylabel = 'Probability'

    plt.title(title)
    plt.hist(x, normed=normed, bins=bins)
    plt.ylabel(ylabel)
    plt.show()

    print ("Done plot_hist: " + title)


def plot_line(x, title='Line', ylabel=None):
    print ("Plot: " + title)

    if ylabel is None:
        ylabel = 'Value'

    plt.plot(x)
    plt.ylabel(ylabel)
    plt.show()

    print("Done plot_line: " + title)


def show_xy_dist_hist(metric_results, y, loader, precent):
    similar_images_metric_results = metric_results[:int(len(metric_results)*precent), :]
    threshold = similar_images_metric_results[len(similar_images_metric_results)-1, 2]

    y_org = loader.y_inverse_transform(y)
    xx1 = y_org[similar_images_metric_results[:, 0].astype(int), 0]
    yy1 = y_org[similar_images_metric_results[:, 0].astype(int), 1]
    xx2 = y_org[similar_images_metric_results[:, 1].astype(int), 0]
    yy2 = y_org[similar_images_metric_results[:, 1].astype(int), 1]
    dist = np.sqrt((xx1 - xx2) ** 2 + (yy1 - yy2) ** 2)
    plot_hist(dist, False, 30, title='Distances hist of %s/%s (%s%%) pairs with IOU>%s' %
                                     (len(dist), len(metric_results), 100*precent, threshold), ylabel='Pairs Number')


def all_to_all(smooth_kernel=None):
    loader = meshNet_loader.DataLoader(data_dir=data_dir, x_range=(0, 1), part_of_data=1.0/20)
    # , pkl_cache_file_$path='/home/moti/cg/project/meshNet/sessions_outputs/mesh_data.pkl')

    x = np.concatenate((loader.x_train, loader.x_test))
    y = np.concatenate((loader.y_train, loader.y_test))
    # file_urls = np.concatenate((loader.file_urls_train, loader.file_urls_test))

    # visualize.show_data(x, border_size=2, bg_color=(255, 0, 0))

    expected_pairs_num = int(comb(x.shape[0], 2))
    print("Expected pairs number is %s" % expected_pairs_num)
    metric_results = np.empty((expected_pairs_num, 3), dtype=np.float32)

    count = 0
    start_time = time.time()
    for i in xrange(x.shape[0]):
        for j in xrange(i + 1, x.shape[0]):
            metric = calc_iou(x[i], x[j], smooth_kernel)

            metric_results[count] = np.array([i, j, metric])
            count += 1

            if count % 10000 == 0:
                end_time = time.time()
                print("Calculated IOU for sample %s/%s, last 10000 took %d seconds" %
                      (count, expected_pairs_num, end_time - start_time))
                start_time = time.time()

    # Sort results by 3rd column - metric results
    metric_results = metric_results[(-metric_results[:, 2]).argsort()]

    print("Max IOU %s at index %s" % (np.max(metric_results[:, 2]), np.argmax(metric_results[:, 2])))
    plot_hist(metric_results[:, 2], False, 30, title='IOU of %s pairs' % len(metric_results[:, 2]), ylabel='IOU Result')
    plot_line(metric_results[:, 2], title='IOU of %s pairs' % len(metric_results[:, 2]))

    show_xy_dist_hist(metric_results, y, loader, 0.1)
    show_xy_dist_hist(metric_results, y, loader, (1.0/5000)/7)

    pairs_num = 50
    image_pairs = np.empty((pairs_num*2, x.shape[1], x.shape[2], 1), dtype=x.dtype)
    for i in xrange(50):
        img1_idx = int(metric_results[i, 0])
        img2_idx = int(metric_results[i, 1])

        image_pairs[i*2, :] = x[img1_idx]
        image_pairs[i*2 + 1, :] = x[img2_idx]

        print("pair #" + str(i) + ", IOU: " + str(metric_results[i, 2]))
        print(loader.y_inverse_transform(y[img1_idx]))
        print(loader.y_inverse_transform(y[img2_idx]))
        print("")

    visualize.show_data(image_pairs, h_axis_num=2, border_size=3, bg_color=(255, 0, 0),
                        write='/home/moti/cg/project/meshNet/sessions_outputs/')


def n_to_all(smooth_kernel=None):
    n = 100

    loader = meshNet_loader.DataLoader(data_dir=data_dir, x_range=(0, 1), part_of_data=1.0)

    x = np.concatenate((loader.x_train, loader.x_test))
    y = np.concatenate((loader.y_train, loader.y_test))
    # file_urls = np.concatenate((loader.file_urls_train, loader.file_urls_test))

    # Take n samples off of x an y
    rand_set = random.sample(xrange(x.shape[0]), n)
    print(x.shape)
    print(y.shape)
    x_small = x[rand_set, :]
    y_small = y[rand_set, :]
    x = np.delete(x, rand_set, axis=0)
    y = np.delete(y, rand_set, axis=0)
    print(x.shape)
    print(y.shape)

    expected_pairs_num = n * x.shape[0]
    print("Expected pairs number is %s" % expected_pairs_num)
    metric_results = np.empty((expected_pairs_num, 3), dtype=np.float32)

    count = 0
    start_time = time.time()
    for i in xrange(x_small.shape[0]):
        for j in xrange(x.shape[0]):
            # metric = calc_ssd(x[i], x[j])
            metric = calc_iou(x_small[i], x[j], smooth_kernel)

            metric_results[count] = np.array([i, j, metric])
            count += 1

            if count % 10000 == 0:
                end_time = time.time()
                print("Calculated IOU for sample %s/%s, last 10000 took %d seconds" %
                      (count, expected_pairs_num, end_time - start_time))
                start_time = time.time()

    # Sort results by 3rd column - metric results
    metric_results = metric_results[(-metric_results[:, 2]).argsort()]

    print("Max IOU %s" % np.max(metric_results[:, 2]))
    plot_hist(metric_results[:, 2], False, 30, title='IOU of %s pairs' % len(metric_results[:, 2]), ylabel='IOU Result')
    plot_line(metric_results[:, 2], title='IOU of %s pairs' % len(metric_results[:, 2]))

    def show_xy_dist_hist_n_to_all(metric_results, y_small, y, loader, precent):
        similar_images_metric_results = metric_results[:int(len(metric_results) * precent), :]
        threshold = similar_images_metric_results[len(similar_images_metric_results) - 1, 2]

        y_small_org = loader.y_inverse_transform(y_small)
        y_org = loader.y_inverse_transform(y)
        xx1 = y_small_org[similar_images_metric_results[:, 0].astype(int), 0]
        yy1 = y_small_org[similar_images_metric_results[:, 0].astype(int), 1]
        xx2 = y_org[similar_images_metric_results[:, 1].astype(int), 0]
        yy2 = y_org[similar_images_metric_results[:, 1].astype(int), 1]
        dist = np.sqrt((xx1 - xx2) ** 2 + (yy1 - yy2) ** 2)
        plot_hist(dist, False, 30, title='Distances hist of %s/%s (%s%%) pairs with IOU>%s' %
                                         (len(dist), len(metric_results), 100 * precent, threshold),
                  ylabel='Pairs Number')

    show_xy_dist_hist_n_to_all(metric_results, y_small, y, loader, 0.1)
    show_xy_dist_hist_n_to_all(metric_results, y_small, y, loader, 0.03)

    pairs_num = 50
    image_pairs = np.empty((pairs_num*2, x.shape[1], x.shape[2], 1), dtype=x.dtype)
    for i in xrange(50):
        img1_idx = int(metric_results[i, 0])
        img2_idx = int(metric_results[i, 1])

        image_pairs[i*2, :] = x_small[img1_idx]
        image_pairs[i*2 + 1, :] = x[img2_idx]

        print("pair #" + str(i) + ", IOU: " + str(metric_results[i, 2]))
        print(loader.y_inverse_transform(y_small[img1_idx]))
        print(loader.y_inverse_transform(y[img2_idx]))
        print("")

    visualize.show_data(image_pairs, h_axis_num=2, border_size=3, bg_color=(255, 0, 0),
                        write='/home/moti/cg/project/meshNet/sessions_outputs/')


def main():
    # all_to_all()
    # n_to_all()

    all_to_all(7)
    # n_to_all(7)

    #     cv2.ims   how("img2", x[img2_idx])
    #     key = cv2.waitKey(0) & 0xFF
    #     if key == 27:
    #         break

    print("Done")

if __name__ == '__main__':
    main()
