import os
import time

import numpy as np
import numpy.ma as ma
from scipy.misc import imsave
from mpl_toolkits.axes_grid1 import make_axes_locatable
import matplotlib.pyplot as plt
import matplotlib.cm as cm

import cv2
from sklearn.preprocessing import StandardScaler

import consts
import utils


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


def show_data(x, scaler=None, offset=0, images_num_per_axis=None, title="show_data"):
    key = None
    done = False

    # TODO: Use constants instead of fixed number for dimensions
    # img_rows, img_cols, img_channels = keras_utils.parse_batch_shape(x)
    # The problem with this solution is that the batches are usually still in TF style during pre-processing.

    img_rows = x.shape[1]
    img_cols = x.shape[2]
    img_channels = x.shape[3]

    if images_num_per_axis is None:
        images_num_per_axis = 600 // img_rows

    # (images_num_per_axis-1) is added form grid lines
    images = np.zeros((img_rows * images_num_per_axis + images_num_per_axis-1,
                       img_cols * images_num_per_axis + images_num_per_axis-1,
                       img_channels), np.uint8)

    if scaler is not None:
        scale_img = scaler.scale_.reshape(img_rows, img_cols, img_channels)
        mean_img = scaler.mean_.reshape(img_rows, img_cols, img_channels)
    else:
        scale_img = np.full(x.shape[1:], 1, np.float32)
        mean_img = np.zeros(x.shape[1:], np.float32)

    while key != 27 and not done:  # 27 is Esc key
        for row in range(images_num_per_axis):
            for col in range(images_num_per_axis):
                cur_idx = offset + row * images_num_per_axis + col
                if cur_idx >= x.shape[0]:
                    done = True
                    break

                cur_img = (x[cur_idx, ...] * scale_img + mean_img).astype('uint8')
                images[row * img_rows + row:row * img_rows + img_rows + row, col * img_cols + col:col * img_cols + img_cols + col, :] = cur_img

        # TODO: Change window title to show current images range
        cv2.imshow(title, images)
        print("Press Esc to exit or any other key to continue")
        key = cv2.waitKey(0) & 0xFF

        offset += images_num_per_axis * images_num_per_axis
        images.fill(0)

        # TODO: Close window automatically when done
        # cv2.destroyWindow(title)


def visualize_history(history, sess_info):
    try:
        target_names = ['training dataset', 'validation dataset']
        fig = plt.figure()
        fig.suptitle(sess_info.title)

        if "acc" in history.history.keys():
            ax = fig.add_subplot(121)
        else:
            ax = fig.add_subplot(111)

        ax.plot(history.epoch, history.history['loss'], 'r', label=target_names[0])
        ax.plot(history.epoch, history.history['val_loss'], 'g', label=target_names[1])
        ax.legend()
        ax.set_title('Loss')

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
