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

# Script config
title = "meshNet"
sess_info = utils.SessionInfo(title)

#sessions_outputs = '/home/moti/cg/project/sessions_outputs'
sessions_outputs = '/mnt/SSD1/moti/project/sessions_outputs'

# data_dir = os.path.join(sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200/')
# data_dir = os.path.join(sessions_outputs, 'berlinRoi_4400_5500_800_800Grid400/')
# data_dir = os.path.join(sessions_outputs, 'berlinRoi_4400_5500_800_800Grid800/')
# data_dir = os.path.join(sessions_outputs, 'berlin_angleOnly_4950_5850/')
# data_dir = os.path.join(sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200_Full/')
data_dir = os.path.join(sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200_FullImage')

# data_dir = os.path.join(sessions_outputs, 'berlin_many_angels_few_xys/-1520.15_1422.77/')
# data_dir = os.path.join(sessions_outputs, 'berlin_onlyPos_grid50/')
# data_dir = os.path.join(sessions_outputs, 'berlin_grid50/')
# data_dir = os.path.join(sessions_outputs, 'project_2017_09_06-12_40_19-grid_20/')
# data_dir = os.path.join(sessions_outputs, 'project_2017_09_06-21_41_07-grid_40/')
train_dir = os.path.join(data_dir, 'train')
# test_dir = None
test_dir = os.path.join(data_dir, 'test')

# weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_10_10-14_13_59-25Epochs-Grid20-almost-PoseNet/hdf5/meshNet_weights.e024-vloss0.3175.hdf5'
# weights_filename = '/home/arik/Desktop/moti/project/meshNet/sessions_outputs/meshNet_2017_11_21-09_45_32/hdf5/meshNet_weights.e047-vloss0.5336.hdf5'
# weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_11_24-09_37_32_60Epochs_Berlin_Grid50/hdf5/meshNet_weights.e038-loss0.54771-vloss0.5626.hdf5'
# weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_11_28-13_53_12-100Epochs_Berlin_ROI_Grid200_NoAngles/hdf5/meshNet_weights.e093-loss0.07357-vloss0.5875.hdf5'

# 200
# weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_12_06-12_40_01-100Epochs_Grid200_Batch4/hdf5/meshNet_weights.e098-loss0.07383-vloss0.4329.hdf5'

# 400
# weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_12_06-13_24_29-100Epochs_Grid400_Batch8/hdf5/meshNet_weights.e085-loss0.04812-vloss0.1097.hdf5'
# weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_12_06-13_24_29-100Epochs_Grid400_Batch8/hdf5/meshNet_weights.e088-loss0.05448-vloss0.0942.hdf5'

# angle only
# weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_12_13-12_40_52-100Epochs_anglesOnly_Batch8/hdf5/meshNet_weights.e090-loss0.00323-vloss0.0023.hdf5'

# 200 - FULL
weights_filename = '/home/moti/cg/project/meshNet/sessions_outputs/meshNet_2017_12_20-10_30_58/hdf5/meshNet_weights.e017-loss0.21648-vloss0.3123.hdf5'

use_pickle = False
render_to_screen = False
test_only = False
load_weights = False
initial_epoch = 0  # Should have the weights Epoch number + 1
if test_only:
    load_weights = True

# Training options
batch_size = 32
save_best_only = True

x_type = 'edges_on_faces'  # 'edges', 'gauss_blur_15', 'edges_on_faces'
y_type = 'quaternion'  # 'angle', 'quaternion', 'matrix'

debug_level = 0

if debug_level == 0:    # No Debug
    part_of_data = 1.0
    epochs = 100
elif debug_level == 1:  # Medium Debug
    part_of_data = 0.3
    epochs = 15
elif debug_level == 2:  # Full Debug
    part_of_data = 100
    epochs = 2
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


# This calcs the L2 Loss. Notice the difference from calc_stats - No sqrt() per element
# def calc_loss(y, y_pred):
#     l2_loss = np.mean(np.square(y - y_pred))
#     return l2_loss


def print_results(y, y_pred, file_urls, xy_error):
    for i in xrange(y_pred.shape[0]):
        print("Sample #%s. File: %s" % (i, file_urls[i]))
        print ("Ground-Truth: %s, Prediction: %s, xy_error: %s" % (y[i], y_pred[i], xy_error[i]))
        print ("")


def calc_stats(loader, y, y_pred, normalized=False, dataset_name='dataset'):
    if not normalized:
        y = loader.y_inverse_transform(y)
        y_pred = loader.y_inverse_transform(y_pred)

    print("%s errors..." % dataset_name)
    xy_error = utils.xy_dist(y[:, 0:2], y_pred[:, 0:2])
    print("%s xy error. Mean %s, std %s" % (dataset_name, np.mean(xy_error), np.std(xy_error)))
    angle_error = utils.angle_l2_err(y[:, 2:4], y_pred[:, 2:4], normalized=normalized)
    print("%s angle error. Mean %s, std %s" % (dataset_name, np.mean(angle_error), np.std(angle_error)))

    return xy_error, angle_error


# visualize.view_prediction(data_dir, loader, y_train_pred, xy_error_train, idx=5)
def detailed_evaluation(model, loader, posenet_output=3):
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
    print("Using PoseNet output [%d]" % posenet_output)
    xyz_output = (posenet_output - 1) * 2
    angle_output = xyz_output + 1
    y_train_pred = np.concatenate((y_train_pred[xyz_output], y_train_pred[angle_output]), axis=-1)
    y_test_pred = np.concatenate((y_test_pred[xyz_output], y_test_pred[angle_output]), axis=-1)

    xy_error_train, angle_error_train = calc_stats(loader, loader.y_train, y_train_pred, normalized=False,
                                                   dataset_name='Train')
    xy_error_test, angle_error_test = calc_stats(loader, loader.y_test, y_test_pred, normalized=False,
                                                 dataset_name='Test')

    # for i in xrange(5):
    #    visualize.view_prediction(data_dir, loader, y_train_pred, y_test_pred, idx=i, is_train=True, by_xy=True,
    #                              normalized=False, asc=False, figure_num=i)

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

    if load_weights and use_pickle:
        # pickle_full_path = os.path.join(os.path.dirname(weights_filename), os.path.pardir, 'pickle',
        #                                 sess_info.title + '.pkl')
        # loader = meshNet_loader.DataLoader()
        # loader.load_pickle(pickle_full_path, part_of_data=part_of_data)
        pass
    else:
        loader = meshNet_loader.DataLoader()
        loader.load(train_dir=train_dir,
                    test_dir=test_dir,
                    x_range=(0, 1),
                    x_type=x_type,
                    y_type=y_type,
                    part_of_data=part_of_data)

    # visualize.show_data(loader.x_train, bg_color=(0, 255, 0))

    print("Getting model...")
    image_shape = loader.x_train.shape[1:]
    nb_outs = len(loader.y_train[0])

    # Custom loss is mandatory to take 360 degrees into consideration
    # params = {'image_shape': image_shape, 'nb_outs': nb_outs, 'loss': meshNet_model.meshNet_loss}
    # model, model_name = meshNet_model.reg_2_conv_relu_mp_2_conv_relu_dense_dense(**params)
    # model, model_name = meshNet_model.reg_2_conv_relu_mp_2_conv_relu_dense_dense_bigger(**params)
    # model, model_name = meshNet_model.almost_VGG11_bn(**params)
    import posenet
    params = {'image_shape': image_shape, 'xy_nb_outs': 2, 'rot_nb_outs': nb_outs-2}
    model, model_name = posenet.posenet_train(**params)

    if load_weights:
        meshNet_model.load_model_weights(model, weights_filename)

    model.summary()
    print("")

    print("image_shape:", image_shape)
    print("nb_outs:", nb_outs)
    print("Model name:", model_name)
    print("Model function input arguments:", params)
    print("Batch size:", batch_size)
    print("")

    print("Model params number:", model.count_params())
    print("Model loss type: %s" % model.loss)
    print("Model optimizer:", model.optimizer)
    print("")

    print("x_type:", loader.x_type)
    print("y_type:", loader.y_type)
    print("y_min_max:", loader.y_min_max)
    print("y range:", loader.y_min_max[1] - loader.y_min_max[0])
    print("")

    if not test_only:
        print("Training model...")
        # TODO: Consider adding custom callback to monitor each loss separately and save best epochs accordingly
        callbacks = meshNet_model.get_checkpoint(sess_info, is_classification=False, save_best_only=save_best_only,
                                                 tensor_board=False)

        history = model.fit(loader.x_train, [loader.y_train[:, :2], loader.y_train[:, 2:],
                                             loader.y_train[:, :2], loader.y_train[:, 2:],
                                             loader.y_train[:, :2], loader.y_train[:, 2:]],
                            batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                            validation_data=(loader.x_test,
                                             [loader.y_test[:, :2], loader.y_test[:, 2:],
                                              loader.y_test[:, :2], loader.y_test[:, 2:],
                                              loader.y_test[:, :2], loader.y_test[:, 2:]]),
                            shuffle=True, initial_epoch=initial_epoch)
        # history = model.fit(loader.x_train, loader.y_train, batch_size=batch_size, epochs=epochs,
        # callbacks=callbacks, validation_data = (loader.x_test, loader.y_test), shuffle = True)

        meshNet_model.load_best_weights(model, sess_info)
    else:
        history = None

    test_score = model.evaluate(loader.x_test, [loader.y_test[:, :2], loader.y_test[:, 2:],
                                loader.y_test[:, :2], loader.y_test[:, 2:],
                                loader.y_test[:, :2], loader.y_test[:, 2:]], batch_size=batch_size, verbose=0)
    # test_score = model.evaluate(loader.x_test, loader.y_test, batch_size=batch_size, verbose=0)
    print('Test score:', test_score)

    # if not test_only:
    #     loader.save_pickle(sess_info)

    if history:
        visualize.visualize_history(history, sess_info, render_to_screen)

    # TODO: Choose best output according to test_score
    # detailed_evaluation(model, loader, 1)
    # detailed_evaluation(model, loader, 2)
    detailed_evaluation(model, loader, 3)


if __name__ == '__main__':
    main()
