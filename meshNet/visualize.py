from __future__ import print_function
from __future__ import division

import os
import time

import numpy as np
import numpy.ma as ma
from scipy.misc import imsave
from mpl_toolkits.axes_grid1 import make_axes_locatable

import matplotlib
# Force matplotlib to not use any Xwindows backend.
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2
from sklearn.preprocessing import StandardScaler

import consts
import utils

USE_OPENCV = False


def imshow(win_title, img):
    img_range = (np.min(img), np.max(img))
    if img.dtype in [np.float, np.float32, np.float64] and (img_range[0] < 0 or img_range[1] > 1):
        print("Floating image not in [0, 1]. Converting to [0, 1]...")
        img = utils.min_max_scale(img, img_range, (0, 1))

    if USE_OPENCV:
        cv2.imshow(win_title, img)
        key = cv2.waitKey(0) & 0xFF
        return key
    else:
        # rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB
        if len(img.shape) == 3 and img.shape[2] == 3:
            rgb = np.zeros_like(img)
            rgb[:, :, 0] = img[:, :, 2]
            rgb[:, :, 1] = img[:, :, 1]
            rgb[:, :, 2] = img[:, :, 0]
        else:
            rgb = cv2.cvtColor(img, cv2.COLOR_GRAY2RGB)

        plt.title(win_title)
        plt.imshow(rgb)
        plt.show()
        return 27  # Esc - Will show only 1st window


def multiple_plots(figure_num, nrows, ncols, plot_number):
    plt.figure(figure_num)
    plt.subplot(nrows, ncols, plot_number)


def plot_hist(x, normed, bins, title='Histogram', ylabel=None, show=True, save_path=None):
    print ("Plot: " + title)

    if ylabel is None and normed:
        ylabel = 'Probability'

    plt.title(title)
    plt.hist(x, normed=normed, bins=bins)
    plt.ylabel(ylabel)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    print ("Done plot_hist: " + title)


def plot_2d_hist(x, y, normed, bins, title='Histogram', xlabel=None, ylabel=None, xlim=None, ylim=None, show=True,
                 save_path=None):
    print ("Plot: " + title)

    heatmap, xedges, yedges = np.histogram2d(x, y, normed=normed, bins=bins)
    extent = [xedges[0], xedges[-1], yedges[0], yedges[-1]]

    plt.title(title)
    cax = plt.imshow(heatmap.T, cmap='jet', aspect='auto', extent=extent, origin='lower')
    plt.colorbar(cax)
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if xlim is not None:
        plt.xlim(xlim)
    if ylim is not None:
        plt.ylim(ylim)

    if save_path is not None:
        plt.savefig(save_path)

    if show:
        plt.show()

    print ("Done plot_hist: " + title)


def plot_line(x, title='Line', ylabel=None, show=True):
    print ("Plot: " + title)

    if ylabel is None:
        ylabel = 'Value'

    plt.plot(x)
    plt.ylabel(ylabel)
    if show:
        plt.show()

    print("Done plot_line: " + title)
    return plt


def render_view(pose):
    if len(pose) != 4:
        raise Exception("Pose must be of the form (x,y,yaw,pitch)")

    view_str = "%f %f %f %f %f %f" % (pose[0], pose[1], 9999, pose[2], pose[3], 0)

    print("Calling PROJECT with pose %s" % view_str)

    from subprocess import call
    file_path = '/home/moti/cg/project/meshNet/lastEdgeView.png'
    call(['../project', '../../berlin/berlin.obj', '-single=' + file_path, '-output_dir=temp', '-pose=' + view_str])
    img = cv2.imread(file_path)
    # utils.rm_file(file_path)
    return img


class OrthoData:
    def __init__(self, data_dir):
        self.filepath = os.path.join(data_dir, 'map_ortho_data.txt')

        self.left, self.right, self.bottom, self.top, self.neardist, self.fardist = self.read()

    def read(self):
        with open(self.filepath) as f:
            line = f.readline()

        ortho_data_str = line[line.find('=(') + 2:len(line) - 1]

        ortho_data_list = []
        for num in ortho_data_str.split(','):
            ortho_data_list.append(float(num))

        return tuple(ortho_data_list)

    def convert_world_point_to_map(self, p, map_shape):
        map_x = (p[0] - self.left) / (self.right - self.left) * map_shape[1]
        map_y = (p[1] - self.bottom) / (self.top - self.bottom) * map_shape[0]

        return int(round(map_x)), int(round(map_y))


def get_map_view(data_dir, p1, p2, roi=None):
    map_file_path = os.path.join(data_dir, 'gSamplesMap.png')
    img_map = cv2.imread(map_file_path)

    orth_data = OrthoData(data_dir)
    map_p1 = orth_data.convert_world_point_to_map(p1, img_map.shape)
    map_p2 = orth_data.convert_world_point_to_map(p2, img_map.shape)

    # cv_azul_color = (255, 255, 0)
    # cv_green_color = (0, 255, 0)
    cv_red_color = (0, 0, 255)
    cv_yellow_color = (45, 205, 243)

    cv2.circle(img_map, center=map_p1, radius=12, color=cv_yellow_color, thickness=cv2.FILLED)
    cv2.circle(img_map, center=map_p2, radius=12, color=cv_red_color, thickness=cv2.FILLED)

    if roi is not None:
        left, top, width, height = roi
        img_map = img_map[top:top + height, left:left + width]

    return img_map


# make view_prediction() local:
# import cv2
# import matplotlib.pyplot as plt
# from visualize import render_view
# from visualize import get_map_view
# from visualize import multiple_plots
def view_prediction(data_dir, roi, loader, y_train_pred, y_test_pred, errors_by, idx, is_train=True, normalized=False,
                    asc=True, figure_num=99):
    if is_train:
        y = loader.y_train
        y_pred = y_train_pred
    else:
        y = loader.y_test
        y_pred = y_test_pred

    normalized_errors = np.linalg.norm(y - y_pred, axis=-1)

    if not normalized:
        y = loader.y_inverse_transform(y)
        y_pred = loader.y_inverse_transform(y_pred)

    xy_errors = utils.xy_dist(y[:, :2], y_pred[:, :2])
    angle_errors = utils.rotation_error(y[:, 2:], y_pred[:, 2:], normalized)

    if errors_by == 'xy':
        errors = xy_errors
    elif errors_by == 'angle':
        errors = angle_errors
    elif errors_by == 'comb':
        errors = normalized_errors
    else:
        raise Exception("Unknown errors_by argument")

    # Convert 'y' to (x,y,yaw,pitch) as this is the current visualization
    # Another possibility would be to convert everything to quaternions and calculate the joint-rotation-angle error
    if y.shape[1] == 4:  # 'angle'
        pass
    elif y.shape[1] == 6:  # 'quaternion'
        y = utils.convert_quaternion_y_to_yaw_pitch(y)
        y_pred = utils.convert_quaternion_y_to_yaw_pitch(y_pred)
    else:
        raise Exception("Only 'angle' and 'quaternion' are currently supported")

    sort_idx = np.argsort(errors) if asc else np.argsort(errors)[::-1]

    y_single = y[sort_idx][idx]
    y_pred_single = y_pred[sort_idx][idx]

    if normalized:
        raise Exception("Normalized is currently not supported")

    img_org = render_view(y_single)
    img_pred = render_view(y_pred_single)
    img_map = get_map_view(data_dir, y_single[:2], y_pred_single[:2], roi)

    upper_part_height = int(0.33333 * img_org.shape[0])
    cv_red_color = (0, 0, 255)
    cv2.line(img_org, (0, upper_part_height), (img_org.shape[1], upper_part_height), cv_red_color, 3)
    cv2.line(img_pred, (0, upper_part_height), (img_org.shape[1], upper_part_height), cv_red_color, 3)

    img_org = cv2.cvtColor(img_org, cv2.COLOR_BGR2RGB)
    img_pred = cv2.cvtColor(img_pred, cv2.COLOR_BGR2RGB)

    # The complex line efficient by far (It avoids copying the sorted image array)
    # x_input = loader.x_train[sort_idx][idx]
    x_input = loader.x_train[np.arange(len(loader.x_train))[sort_idx][idx]]

    img_train = cv2.cvtColor(x_input, cv2.COLOR_GRAY2RGB)
    img_map = cv2.cvtColor(img_map, cv2.COLOR_BGR2RGB)

    np.set_printoptions(formatter={'float_kind': lambda x: "%.2f" % x})

    multiple_plots(figure_num, 2, 2, 1)
    plt.imshow(img_org, interpolation='bilinear')
    plt.title('Original %s' % y_single)

    multiple_plots(figure_num, 2, 2, 2)
    plt.imshow(img_pred, interpolation='bilinear')
    plt.title('Estimation %s' % y_pred_single)

    np.set_printoptions()  # Reset

    multiple_plots(figure_num, 2, 2, 3)
    plt.imshow(img_train)
    plt.title('NN Input')

    multiple_plots(figure_num, 2, 2, 4)
    plt.imshow(img_map)
    plt.title('Map')

    angle_diff = utils.angle_diff(y_single[2:], y_pred_single[2:])
    plt.suptitle("idx = %i/%i (%s,%s), errors: xy=%s, yaw=%s, pitch=%s, angle_l2=%s, comb=%s" %
                 (idx if asc else len(errors) - 1 - idx, len(errors) - 1, 'asc' if asc else 'desc', errors_by,
                  xy_errors[sort_idx][idx],
                  angle_diff[0], angle_diff[1], angle_errors[sort_idx][idx],
                  normalized_errors[sort_idx][idx]))  # , fontsize=16)

    plt.show()


def show_predictions(x, y, prediction, file_urls, scaler, encoder, resize_factor=4, title="show_image"):
    key = None

    if scaler is None:
        scaler = StandardScaler()
        scaler.mean_ = np.zeros(x.shape[1:], np.float32).flatten() * 128.
        scaler.scale_ = np.full(x.shape[1:], 255., dtype=np.float32).flatten()

    orig_shape = x.shape
    x_flatten = x.reshape(len(x), -1)
    x_flatten = scaler.inverse_transform(x_flatten)
    x = x_flatten.reshape(orig_shape)
    x = x.astype('uint8')

    i = 0
    while key != 27:
        if key == ord('n') and i < x.shape[0] - 1:
            i += 1
        elif key == ord('p') and i > 0:
            i -= 1
        elif key == ord('b'):
            resize_factor *= 2
        elif key == ord('s'):
            resize_factor /= 2
        elif key == ord('o'):
            cv2.imshow(title, cv2.imread(file_urls[i]))
            cv2.waitKey(0)

        img = utils.get_image_from_batch(x, i)
        show_img = cv2.resize(img, (0, 0), fx=resize_factor, fy=resize_factor)

        text = "class {0} ->".format(encoder.inverse_transform(y[i]))
        cv2.putText(show_img, text, (20, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.4 if img.shape[1] > 30 else 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)
        text = "predicted {0}, factor {1}".format(encoder.inverse_transform(prediction[i]), resize_factor)
        cv2.putText(show_img, text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.4 if img.shape[1] > 30 else 0.35,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.putText(show_img, file_urls[i], (20, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.4 if img.shape[1] > 30 else 0.3,
                    (255, 255, 255), 1, cv2.LINE_AA)
        cv2.imshow(title, show_img)

        print("Current file: " + file_urls[i] +
              " 'n'-next, 'p'-prev, 'b'-bigger, 's'-smaller, 'o'-open original, Esc to Exit")
        key = cv2.waitKey(0) & 255  # For some reason I sometime get a very large number and not the clean ASCII


# FIXME: Add border_color instead of bg_color - low priority
def show_data(x, offset=0, h_axis_num=None, v_axis_num=None, border_size=1, bg_color=(0, 0, 0), write=None):
    key = None
    done = False

    img_rows = x.shape[1]
    img_cols = x.shape[2]
    img_channels = x.shape[3]

    if v_axis_num is None:
        v_axis_num = 600 // img_rows
    if h_axis_num is None:
        h_axis_num = 800 // img_cols

    # (images_num_per_axis-1) is added form grid lines
    images = np.zeros((img_rows * v_axis_num + (v_axis_num - 1) * border_size,
                       img_cols * h_axis_num + (h_axis_num - 1) * border_size,
                       3), np.uint8)

    images[:] = bg_color

    if x.dtype == np.float32:
        mean_img = np.full(x.shape[1:], np.abs(np.min(x)), np.float32)  # Handle e.g. (0, 1) or (-0.5, 0.5)
        scale_img = np.full(x.shape[1:], 255, np.float32)
    else:
        scale_img = np.full(x.shape[1:], 1, np.float32)
        mean_img = np.zeros(x.shape[1:], np.float32)

    while key != 27 and not done:  # 27 is Esc key
        for row in range(v_axis_num):
            for col in range(h_axis_num):
                cur_idx = offset + row * h_axis_num + col
                if cur_idx >= x.shape[0]:
                    done = True
                    break

                cur_img = ((x[cur_idx, ...] + mean_img) * scale_img).astype('uint8')
                if img_channels == 1:
                    cur_img = cv2.cvtColor(cur_img, cv2.COLOR_GRAY2RGB)
                images[row * img_rows + row * border_size:row * img_rows + img_rows + row * border_size,
                       col * img_cols + col * border_size:col * img_cols + img_cols + col * border_size, :] = cur_img

        current_images = str(offset) + "-" + str(offset + v_axis_num * h_axis_num - 1) + "_of_" + str(x.shape[0])
        title = "show_data_" + current_images
        # cv2.namedWindow(title)
        # cv2.moveWindow(title, 50, 50)
        key = imshow(title, images)
        print("Images: " + current_images + ". Press Esc to exit or any other key to continue")

        # key = cv2.waitKey(0) & 0xFF

        if write:
            image_path = os.path.join(write, title + ".png")
            cv2.imwrite(image_path, images)

        offset += v_axis_num * h_axis_num
        images[:] = bg_color
        # cv2.destroyWindow(title)


def visualize_history(history, sess_info, render_to_screen=True):
    try:
        print("History results-")
        print(history.history)
        print("")
        for d in history.history:
            print("%s = %s" % (d, history.history[d]))

        target_names = ['training-set', 'validation-set']
        fig = plt.figure()
        fig.suptitle(sess_info.title)

        if "acc" in history.history.keys():
            ax = fig.add_subplot(121)
        else:
            ax = fig.add_subplot(111)

        ax.plot(history.epoch, history.history['loss'], 'r', label=target_names[0])
        ax.plot(history.epoch, history.history['val_loss'], 'g', label=target_names[1])
        ax.legend()
        ax.set_ylim(ymin=0,
                    ymax=3 * max(history.history['val_loss']))  # Avoid very high values 'loss' might starts with
        ax.set_title('Loss (train [%.2f, %.2f], val [%.2f, %.2f])' %
                     (min(history.history['loss']), max(history.history['loss']),
                      min(history.history['val_loss']), max(history.history['val_loss'])))

        if "acc" in history.history.keys():
            ax = fig.add_subplot(122)
            ax.plot(history.epoch, history.history['acc'], 'r', label=target_names[0])
            ax.plot(history.epoch, history.history['val_acc'], 'g', label=target_names[1])
            ax.legend()
            ax.set_title('Accuracy')

        # Save history plot to disk
        history_plot_fname = sess_info.title + '_history_plot.png'
        history_plot_full_path = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, history_plot_fname)
        plt.savefig(history_plot_full_path)

        if render_to_screen:
            plt.show()
    except Exception as e:
        print("Warning: {}".format(e))


def tensor_2d_to_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to image array
    x *= 255
    x = np.clip(x, 0, 255).astype('uint8')
    return x


def tensor_3d_to_image(x):
    # normalize tensor: center on 0., ensure std is 0.1
    x -= x.mean()
    x /= (x.std() + 1e-5)
    x *= 0.1

    # clip to [0, 1]
    x += 0.5
    x = np.clip(x, 0, 1)

    # convert to RGB array
    x *= 255
    x = x.transpose((1, 2, 0))
    x = np.clip(x, 0, 255).astype('uint8')
    return x


# Works best if creating model only up to the layer we want to visualize.
# Not sure why - something with the derivatives behaves better.
def visualize_layer_by_maximize_gradients_wrt_input(model, layer_name, input_shape,
                                                    number_of_filter_to_display_in_each_axis):
    # get the symbolic outputs of each "key" layer (we gave them unique names)
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    from keras import backend as K

    # Input size for the model
    img_ndim = input_shape[0]
    img_height = input_shape[1]
    img_width = input_shape[2]

    # Input image with which to derive the layer we visualize
    input_img = model.layers[0].input

    # Layer to visualize
    layer_output = layer_dict[layer_name].output
    nb_filters = layer_output._keras_shape[1]
    filter_width = layer_output._keras_shape[2]
    filter_height = layer_output._keras_shape[3]

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    kept_filters = []
    # for filter_index in range(0, 10):
    for filter_index in range(0, nb_filters):
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(layer_output[:, filter_index, :, :])

        # we compute the gradient of the input picture wrt this loss
        grads = K.gradients(loss, input_img)[0]

        # normalization trick: we normalize the gradient
        grads = normalize(grads)

        # this function returns the loss and grads given the input  picture
        iterate = K.function([input_img, K.learning_phase()], [loss, grads])

        # step size for gradient ascent
        step = 1.

        # we start from a gray image with some random noise
        input_img_data = (np.random.random((1, img_ndim, img_width, img_height)) - 0.5) * 20

        # we run gradient ascent for 20 steps
        for i in range(20):
            loss_value, grads_value = iterate([input_img_data, 0])
            input_img_data += grads_value * step

            print('Current loss value:', loss_value)

        # decode the resulting input image
        if loss_value > 0:
            img = tensor_3d_to_image(input_img_data[0])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stitch the best 64 filters on a 8 x 8 grid.
    n = number_of_filter_to_display_in_each_axis

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    print(len(kept_filters))
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for our filters with a 5px margin in between
    margin = 5
    width = n * img_width + (n - 1) * margin
    height = n * img_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, img_ndim))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(img_width + margin) * i: (img_width + margin) * i + img_width,
                             (img_height + margin) * j: (img_height + margin) * j + img_height, :] = img

    # save the result to disk
    if not os.path.isdir('outputs/'):
        os.mkdir('outputs/')
    if stitched_filters.shape[2] == 1:
        stitched_filters = np.repeat(stitched_filters, repeats=3, axis=2)
    imsave('outputs/stitched_filters_%dx%d.png' % (n, n), stitched_filters)


def nice_imshow(ax, data, vmin=None, vmax=None, cmap=None):
    """Wrapper around pl.imshow"""
    if cmap is None:
        cmap = cm.jet
    if vmin is None:
        vmin = data.min()
    if vmax is None:
        vmax = data.max()

    import pylab as pl

    divider = make_axes_locatable(ax)
    cax = divider.append_axes("right", size="5%", pad=0.05)
    im = ax.imshow(data, vmin=vmin, vmax=vmax, interpolation='nearest', cmap=cmap)
    pl.colorbar(im, cax=cax)


def make_mosaic(imgs, nrows, ncols, border=1):
    """
    Given a set of images with all the same shape, makes a
    mosaic with nrows and ncols
    """
    nimgs = imgs.shape[0]
    imshape = imgs.shape[1:]

    mosaic = ma.masked_all((nrows * imshape[0] + (nrows - 1) * border,
                            ncols * imshape[1] + (ncols - 1) * border),
                           dtype=np.float32)

    paddedh = imshape[0] + border
    paddedw = imshape[1] + border
    for i in range(nimgs):
        row = int(np.floor(i / ncols))
        col = i % ncols

        mosaic[row * paddedh:row * paddedh + imshape[0], col * paddedw:col * paddedw + imshape[1]] = imgs[i]
    return mosaic


def visualize_layer_by_plotting_weights(model, layer_name):
    # Works only for 3D layers (such as conv layer on a 1D input)

    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    import pylab as pl

    # Visualize weights
    layer = layer_dict[layer_name]
    w = layer.W.get_value()
    w = np.squeeze(w)
    print("W shape : ", w.shape)

    pl.figure(figsize=(15, 15))
    pl.title('conv1 weights')
    nice_imshow(pl.gca(), make_mosaic(w, 6, 6), cmap='gray')


def visualize_layer_by_input_images(model, layer_name, input_images_t, number_of_filter_to_display_in_each_axis):
    # get the symbolic outputs of each "key" layer (we gave them unique names).
    layer_dict = dict([(layer.name, layer) for layer in model.layers])

    from keras import backend as K

    # Input size for the model
    img_ndim = input_images_t[0].shape[0]
    img_height = input_images_t[0].shape[1]
    img_width = input_images_t[0].shape[2]

    # Input image with which to predict
    input_img = model.layers[0].input

    # Layer to visualize
    layer_output = layer_dict[layer_name].output
    nb_filters = layer_output._keras_shape[1]
    filter_width = layer_output._keras_shape[2]
    filter_height = layer_output._keras_shape[3]

    def normalize(x):
        # utility function to normalize a tensor by its L2 norm
        return x / (K.sqrt(K.mean(K.square(x))) + 1e-5)

    kept_filters = []
    for filter_index in range(0, nb_filters):
        # for filter_index in range(0, 10):
        print('Processing filter %d' % filter_index)
        start_time = time.time()

        # we build a loss function that maximizes the activation
        # of the nth filter of the layer considered
        loss = K.mean(K.square(layer_output[:, filter_index, :, :]))

        # this function returns the loss and grads given the input  picture
        iterate = K.function([input_img, K.learning_phase()], [loss, layer_output])

        loss_value, layer_out = iterate([input_images_t, 0])

        # decode the resulting input image
        if loss_value > 0:
            img = tensor_3d_to_image(layer_out[:, filter_index, :, :])
            kept_filters.append((img, loss_value))
        end_time = time.time()
        print('Filter %d processed in %ds' % (filter_index, end_time - start_time))

    # we will stitch the best 64 filters on a 8 x 8 grid.
    n = number_of_filter_to_display_in_each_axis

    # the filters that have the highest loss are assumed to be better-looking.
    # we will only keep the top 64 filters.
    print(len(kept_filters))
    kept_filters.sort(key=lambda x: x[1], reverse=True)
    kept_filters = kept_filters[:n * n]

    # build a black picture with enough space for our filters with a 5px margin in between
    margin = 5
    width = n * filter_width + (n - 1) * margin
    height = n * filter_height + (n - 1) * margin
    stitched_filters = np.zeros((width, height, img_ndim))

    # fill the picture with our saved filters
    for i in range(n):
        for j in range(n):
            img, loss = kept_filters[i * n + j]
            stitched_filters[(filter_width + margin) * i: (filter_width + margin) * i + filter_width,
                             (filter_height + margin) * j: (filter_height + margin) * j + filter_height, :] = img

    # save the result to disk
    if not os.path.isdir('outputs/'):
        os.mkdir('outputs/')
    if stitched_filters.shape[2] == 1:
        stitched_filters = np.repeat(stitched_filters, repeats=3, axis=2)
    imsave('outputs/stitched_filters_inputImage%dx%d.png' % (n, n), stitched_filters)
