from __future__ import print_function
from __future__ import division

import time
import os

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

weights_filename = "/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_07_10-16_18_06/hdf5/meshNet_weights.e007-vloss0.0540.hdf5"

test_only = False
if test_only:
    load_weights = True
else:
    load_weights = False

# Training options
batch_size = 32
use_xy_loss_weight = True

debug_level = 0

if debug_level == 0:    # No Debug
    part_of_data = 1.0
    nb_epoch = 25
elif debug_level == 1:   # Medium Debug
    part_of_data = 0.3
    nb_epoch = 15
elif debug_level == 2:  # Full Debug
    part_of_data = 1.0 / 60
    nb_epoch = 2
else:
    raise Exception("Invalid debug level " + str(debug_level))


def predict(model, x):
    # Predict each test sample and measure total time

    start_time = time.time()
    predictions = model.predict(x, batch_size=1, verbose=0)
    total_time = time.time() - start_time
    print("Prediction of %d samples took %s seconds. I.e. %s seconds per sample" %
          (x.shape[0], total_time, total_time / x.shape[0]))
    return predictions


def calc_xy_stats(y, y_pred):
    xy_error = np.linalg.norm(y[:, 0:2] - y_pred[:, 0:2], axis=-1)
    xy_error_mean = np.mean(xy_error)
    xy_error_std = np.std(xy_error)

    return xy_error, xy_error_mean, xy_error_std


def print_results(y, y_pred, file_urls, xy_error):
    for i in xrange(y_pred.shape[0]):
        print("Sample #%s. File: %s" % (i, file_urls[i]))
        print ("Ground-Truth: %s, Prediction: %s, xy_error: %s" % (y[i], y_pred[i], xy_error[i]))
        print ("")


def detailed_evaluation(model, loader):
    print("detailed evaluation...")

    print("Predicting train set...")
    y_train_pred = predict(model, loader.x_train)
    print("Predicting test set...")
    y_test_pred = predict(model, loader.x_test)

    # Get Y values de-normalized
    y_train = loader.y_inverse_transform(loader.y_train)
    y_train_pred = loader.y_inverse_transform(y_train_pred)

    y_test = loader.y_inverse_transform(loader.y_test)
    y_test_pred = loader.y_inverse_transform(y_test_pred)

    print("Train errors...")
    xy_error_train, xy_error_mean_train, xy_error_std_train = calc_xy_stats(y_train, y_train_pred)
    # print_results(y_train, y_train_pred, loader.file_urls_train, xy_error_train)
    print("Mean train xy_error: %s" % xy_error_mean_train)
    print("Standard deviation train xy_error: %s" % xy_error_std_train)

    print("Test errors...")
    xy_error_test, xy_error_mean_test, xy_error_std_test = calc_xy_stats(y_test, y_test_pred)
    # print_results(y_test, y_test_pred, loader.file_urls_test, xy_error_test)
    print("Mean test xy_error: %s" % xy_error_mean_test)
    print("Standard deviation test xy_error: %s" % xy_error_std_test)

    visualize.multiple_plots(1, 1, 2, 1)
    visualize.plot_hist(xy_error_train, False, 50, title='Train-set errors on X-Y (%s-samples)' % len(xy_error_train),
                        ylabel='Samples', show=False)
    visualize.multiple_plots(1, 1, 2, 2)
    plt = visualize.plot_hist(xy_error_test, False, 50, title='Test-set errors on X-Y (%s-samples)' % len(xy_error_test),
                              ylabel='Samples', show=True)

    # Save predictions plots to disk
    predictions_plot_fname = sess_info.title + '_predictions_hist.png'
    predictions_plot_full_path = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, predictions_plot_fname)
    plt.savefig(predictions_plot_full_path)

    utils.save_pickle(sess_info, [y_train, y_train_pred, loader.file_urls_train,
                                  xy_error_train, xy_error_mean_train, xy_error_std_train,
                                  y_test, y_test_pred, loader.file_urls_test,
                                  xy_error_test, xy_error_mean_test, xy_error_std_test])


def main():
    logger.Logger(sess_info)
    print("Entered %s" % title)

    loader = meshNet_loader.DataLoader(
        data_dir='/home/moti/cg/project/sessions_outputs/project_2017_07_30-11_27_19',
        pkl_cache_file_path='/home/moti/cg/project/meshNet/sessions_outputs/mesh_data.pkl')
    loader.set_part_of_data(part_of_data)

    # visualize.show_data(loader.x_train)

    print("Getting model...")
    image_shape = loader.x_train.shape[1:]
    nb_outs = len(loader.y_train[0])
    # print("image_shape: %d " % image_shape)
    # print("nb_outs: %d " % nb_outs)

    if use_xy_loss_weight:
        params = {'image_shape': image_shape, 'nb_outs': nb_outs, 'loss': meshNet_model.meshNet_loss}
    else:
        params = {'image_shape': image_shape, 'nb_outs': nb_outs}
    model, model_name = meshNet_model.reg_2_conv_relu_mp_2_conv_relu_dense_dense(**params)

    print("Model name: %s " % model_name)
    print("Model function input arguments:")
    print(params)

    if load_weights:
        meshNet_model.load_model_weights(model, weights_filename)

    print("Model params number: %d " % model.count_params())
    print("Model loss type: %s " % model.loss)
    print("Model optimizer:")
    # print(model.optimizer.get_config())
    print("")

    if not test_only:
        print("Training model...")
        # Saves the model weights after each epoch if the validation loss decreased
        callbacks = meshNet_model.get_checkpoint(sess_info, is_classification=False, tensor_board=False)

        history = model.fit(loader.x_train, loader.y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks,
                            validation_data=(loader.x_test, loader.y_test), shuffle=True)
    else:
        history = None

    test_score = model.evaluate(loader.x_test, loader.y_test, verbose=0)
    print('Test score:', test_score)

    if history:
        visualize.visualize_history(history, sess_info)

    detailed_evaluation(model, loader)

if __name__ == '__main__':
    main()
