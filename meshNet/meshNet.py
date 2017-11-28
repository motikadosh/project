from __future__ import print_function
from __future__ import division

import time
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np

# Freeze seed - Must be called before import of Keras!
seed = 12345
np.random.seed(seed)
print("Seed state - {seed}".format(seed=seed))

import meshNet_loader
import utils
import logger
import meshNet_model
import visualize
import consts

# Not used in code, for easier interactive debugging
# import matplotlib.pyplot as plt

# from keras import backend as keras_backend
# import tensorflow as tf

# from models import CNN
# from models.layers.spatial_transform import SpatialTransformFreezeController
# from utilities import Logger, visualization, keras_utils, preprocessing, utils, pb_models_data, consts

# Script config
title = "meshNet"
sess_info = utils.SessionInfo(title)

data_dir = '/home/moti/cg/project/sessions_outputs/berlin_many_angels_few_xys/-1520.15_1422.77/'
# data_dir = '/home/arik/Desktop/moti/project/sessions_outputs/berlin_onlyPos_grid50/'
# data_dir = '/home/arik/Desktop/moti/project/sessions_outputs/berlin_grid50/'
# data_dir = '/home/arik/Desktop/moti/project/sessions_outputs/project_2017_09_06-12_40_19-grid_20/'
# data_dir = '/home/moti/cg/project/sessions_outputs/project_2017_09_06-21_41_07-grid_40/'
train_dir = os.path.join(data_dir, 'train')
test_dir = None
# test_dir = os.path.join(data_dir, 'test')

# weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_10_10-14_13_59-25Epochs-Grid20-almost-PoseNet/hdf5/meshNet_weights.e024-vloss0.3175.hdf5'
# weights_filename = '/home/arik/Desktop/moti/project/meshNet/sessions_outputs/meshNet_2017_11_21-09_45_32/hdf5/meshNet_weights.e047-vloss0.5336.hdf5'
weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_11_24-09_37_32_60Epochs_Berlin_Grid50/hdf5/meshNet_weights.e038-loss0.54771-vloss0.5626.hdf5'


render_to_screen = False
test_only = False
load_weights = False
initial_epoch = 0  # Should have the weights Epoch number + 1
if test_only:
    load_weights = True

# Training options
batch_size = 64
save_best_only = True

debug_level = 0

if debug_level == 0:    # No Debug
    part_of_data = 1.0
    nb_epoch = 60
elif debug_level == 1:  # Medium Debug
    part_of_data = 0.3
    nb_epoch = 15
elif debug_level == 2:  # Full Debug
    part_of_data = 100
    nb_epoch = 3
else:
    raise Exception("Invalid debug level " + str(debug_level))

single_spot = False
if single_spot:
    import warnings
    warnings.warn('Running single spot')

    train_dir = '/home/moti/cg/project/sessions_outputs/project_2017_09_06-12_40_19-grid_20/train/-2.34907_7.04721'
    test_dir = None
    test_only = False
    load_weights = False
    part_of_data = 1.0
    nb_epoch = 100


def predict(model, x):
    # Predict each test sample and measure total time

    start_time = time.time()
    predictions = model.predict(x, batch_size=1, verbose=0)
    total_time = time.time() - start_time
    print("Prediction of %d samples took %s seconds. I.e. %s seconds per sample" %
          (x.shape[0], total_time, total_time / x.shape[0]))
    return predictions


# This calcs the L2 Loss. Notice the difference from calc_stats - No sqrt() per element
# def calc_loss(y, y_pred):
#     l2_loss = np.mean(np.square(y - y_pred))
#     return l2_loss


def calc_angle_diff(a, b):
    # Return a signed difference between two angles
    # I.e. the minimum distance from src(a) to dst(b) - consider counter-clockwise as positive
    return (a - b + 180) % 360 - 180


def calc_angle_stats(y, y_pred):
    err = np.linalg.norm(calc_angle_diff(y, y_pred), axis=-1)
    err_mean = np.mean(err)
    err_std = np.std(err)

    return err, err_mean, err_std


def calc_angle_stats_unnormalized(y, y_pred):
    err = np.linalg.norm(calc_angle_diff(y*360.0, y_pred*360.0)/360.0, axis=-1)
    err_mean = np.mean(err)
    err_std = np.std(err)

    return err, err_mean, err_std


def calc_xy_stats(y, y_pred):
    err = np.linalg.norm(y - y_pred, axis=-1)
    err_mean = np.mean(err)
    err_std = np.std(err)

    return err, err_mean, err_std


def print_results(y, y_pred, file_urls, xy_error):
    for i in xrange(y_pred.shape[0]):
        print("Sample #%s. File: %s" % (i, file_urls[i]))
        print ("Ground-Truth: %s, Prediction: %s, xy_error: %s" % (y[i], y_pred[i], xy_error[i]))
        print ("")


def calc_stats(loader, y, y_pred, normalized=False, dataset_name='dataset'):
    angle_stats_fn = calc_angle_stats_unnormalized

    if not normalized:
        y = loader.y_inverse_transform(y)
        y_pred = loader.y_inverse_transform(y_pred)
        angle_stats_fn = calc_angle_stats

    print("%s errors..." % dataset_name)
    xy_error, xy_error_mean, xy_error_std = calc_xy_stats(y[:, 0:2], y_pred[:, 0:2])
    # print_results(y, y_pred, file_urls, xy_error)
    print("%s xy error. Mean %s, std %s" % (dataset_name, xy_error_mean, xy_error_std))
    angle_error, angle_error_mean, angle_error_std = angle_stats_fn(y[:, 2:4], y_pred[:, 2:4])
    print("%s angle error. Mean %s, std %s" % (dataset_name, angle_error_mean, angle_error_std))

    return xy_error, xy_error_mean, xy_error_std, angle_error, angle_error_mean, angle_error_std


def see_view(loader, x, y, idx):
    visualize.imshow("see_view", x[idx])

    pose = loader.y_inverse_transform(y[idx])
    view_str = "%f %f %f %f %f %f" % (pose[0], pose[1], 9999, pose[2], pose[3], 0)
    print("Calling PROJECT with pose %s" % view_str)

    from subprocess import call
    call(['../project', '../../berlin/berlin.obj', '-pose=' + view_str])


def detailed_evaluation(model, loader):
    print("detailed evaluation...")

    print("Predicting train set...")
    y_train_pred = predict(model, loader.x_train)
    print("Predicting test set...")
    y_test_pred = predict(model, loader.x_test)

    # print("Sanity - Should have identical result to the network loss...")
    # l2_train_loss = calc_loss(loader.y_test, y_test_pred)
    # print("Train l2 loss", l2_train_loss)
    # l2_test_loss = calc_loss(loader.y_test, y_test_pred)
    # print("Test l2 loss", l2_test_loss)

    # PoseNet Fix
    y_train_pred = np.concatenate((y_train_pred[4], y_train_pred[5]), axis=-1)
    y_test_pred = np.concatenate((y_test_pred[4], y_test_pred[5]), axis=-1)

    xy_error_train, xy_error_mean_train, xy_error_std_train, \
        angle_error_train, angle_error_mean_train, angle_error_std_train = \
        calc_stats(loader, loader.y_train, y_train_pred, normalized=False, dataset_name='Train')

    xy_error_test, xy_error_mean_test, xy_error_std_test, \
        angle_error_test, angle_error_mean_test, angle_error_std_test = \
        calc_stats(loader, loader.y_test, y_test_pred, normalized=False, dataset_name='Test')

    plots_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir)
    hist_fname = sess_info.title + '_predictions_err_hist.png'
    train_2d_hist_fname = sess_info.title + '_predictions_err_2d_hist.png'

    max_xy_error = max(np.max(xy_error_test), np.max(xy_error_train))

    try:
        # Plot 2d heatmap histograms of the errors
        visualize.multiple_plots(1, 1, 2, 1)
        visualize.plot_2d_hist(xy_error_test, angle_error_test, False, (50, 50), title='Test Err 2D Histogram',
                               xlabel='XY err', ylabel='Angle err', xlim=[0, max_xy_error], ylim=[0, 180], show=False)
        visualize.multiple_plots(1, 1, 2, 2)
        visualize.plot_2d_hist(xy_error_train, angle_error_train, False, (50, 50), title='Train Err 2D Histogram',
                               xlabel='XY err', ylabel='Angle err', xlim=[0, max_xy_error], ylim=[0, 180],
                               show=render_to_screen, save_path=os.path.join(plots_dir, train_2d_hist_fname))

        # Plot 1D histograms of the errors
        visualize.multiple_plots(2, 2, 2, 1)
        visualize.plot_hist(xy_error_train, False, 50, title='Train XY err(%s-samples)' % len(xy_error_train),
                            ylabel='Samples', show=False)
        visualize.multiple_plots(2, 2, 2, 2)
        visualize.plot_hist(xy_error_test, False, 50, title='Test XY err(%s-samples)' % len(xy_error_test),
                            ylabel='Samples', show=False)

        visualize.multiple_plots(2, 2, 2, 3)
        visualize.plot_hist(angle_error_train, False, 50, title='Train angle err(%s-samples)' %
                            len(angle_error_train), ylabel='Samples', show=False)
        visualize.multiple_plots(2, 2, 2, 4)
        visualize.plot_hist(angle_error_test, False, 50, title='Test angle err(%s-samples)' %
                            len(angle_error_test), ylabel='Samples', show=render_to_screen,
                            save_path=os.path.join(plots_dir, hist_fname))
    except Exception as e:
        print("Warning: {}".format(e))


def main():
    logger.Logger(sess_info)
    print("Entered %s" % title)

    if load_weights:
        pickle_full_path = os.path.join(os.path.dirname(weights_filename), os.path.pardir, 'pickle',
                                        sess_info.title + '.pkl')
        loader = meshNet_loader.DataLoader()
        loader.load_pickle(pickle_full_path, part_of_data=part_of_data)
    else:
        loader = meshNet_loader.DataLoader()
        loader.load(sess_info.title,
                    train_dir=train_dir,
                    test_dir=test_dir,
                    x_range=(0, 1),
                    directional_gauss_blur=None,
                    part_of_data=part_of_data)

    # visualize.show_data(loader.x_train, bg_color=(0, 255, 0))

    print("Getting model...")
    image_shape = loader.x_train.shape[1:]
    nb_outs = len(loader.y_train[0])
    # print("image_shape: %d " % image_shape)
    # print("nb_outs: %d " % nb_outs)

    # Custom loss is mandatory to take 360 degrees into consideration
    # params = {'image_shape': image_shape, 'nb_outs': nb_outs, 'loss': meshNet_model.meshNet_loss}
    # model, model_name = meshNet_model.reg_2_conv_relu_mp_2_conv_relu_dense_dense(**params)
    # model, model_name = meshNet_model.reg_2_conv_relu_mp_2_conv_relu_dense_dense_bigger(**params)
    # model, model_name = meshNet_model.almost_VGG11_bn(**params)
    import posenet
    params = {'image_shape': image_shape, 'xy_nb_outs': 2, 'cyc_nb_outs': 2}
    model, model_name = posenet.posenet_train(**params)

    print("Model name: %s " % model_name)
    print("Model function input arguments:", params)
    print("Batch size: %i" % batch_size)

    if load_weights:
        meshNet_model.load_model_weights(model, weights_filename)

    print("Model params number: %d " % model.count_params())
    print("Model loss type: %s " % model.loss)
    model.summary()
    print("")

    if not test_only:
        print("Training model...")
        # Saves the model weights after each epoch if the validation loss decreased
        callbacks = meshNet_model.get_checkpoint(sess_info, is_classification=False, save_best_only=save_best_only,
                                                 tensor_board=False)

        history = model.fit(loader.x_train, [loader.y_train[:, 0:2], loader.y_train[:, 2:4],
                                             loader.y_train[:, 0:2], loader.y_train[:, 2:4],
                                             loader.y_train[:, 0:2], loader.y_train[:, 2:4]],
                            batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks,
                            validation_data=(loader.x_test,
                                             [loader.y_test[:, 0:2], loader.y_test[:, 2:4],
                                              loader.y_test[:, 0:2], loader.y_test[:, 2:4],
                                              loader.y_test[:, 0:2], loader.y_test[:, 2:4]]),
                            shuffle=True, initial_epoch=initial_epoch)
        # history = model.fit(loader.x_train, loader.y_train, batch_size=batch_size, epochs=nb_epoch,
        # callbacks=callbacks, validation_data = (loader.x_test, loader.y_test), shuffle = True)

        meshNet_model.load_best_weights(model, sess_info)
    else:
        history = None

    # TODO: Run test and detailed EVAL on best EPOCH - Minimum val loss
    test_score = model.evaluate(loader.x_test, [loader.y_test[:, 0:2], loader.y_test[:, 2:4],
                                loader.y_test[:, 0:2], loader.y_test[:, 2:4],
                                loader.y_test[:, 0:2], loader.y_test[:, 2:4]], batch_size=1, verbose=0)
    # test_score = model.evaluate(loader.x_test, loader.y_test, batch_size=1, verbose=0)
    print('Test score:', test_score)

    if not test_only:
        loader.save_pickle(sess_info)

    detailed_evaluation(model, loader)

    if history:
        visualize.visualize_history(history, sess_info, render_to_screen)


if __name__ == '__main__':
    main()
