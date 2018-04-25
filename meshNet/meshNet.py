from __future__ import print_function
from __future__ import division

import time
import os

model_type = 'resnet'  # 'posenet'/'resnet'/'fc'

multi_gpu = False
if not multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"

import numpy as np
np.set_printoptions(precision=4, suppress=True)

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

if os.path.expanduser('~') == '/home/moti':
    data_sessions_outputs = '/home/moti/cg/project/sessions_outputs'
    model_sessions_outputs = '/home/moti/cg/project/meshNet/sessions_outputs'
else:
    data_sessions_outputs = '/mnt/SSD1/moti/project/sessions_outputs'
    model_sessions_outputs = '/mnt/SSD1/moti/project/meshNet/sessions_outputs'

# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200/')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid400/')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid800/')
# data_dir = os.path.join(data_sessions_outputs, 'berlin_angleOnly_4950_5850/')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200_Full/')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200_FullImage')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_5000_3000_800_800_GridStep_20')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800_GridStep_40')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800_GridStep_10_train')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_3000_4800_1600_1600_GridStep_20')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_3000_3000_1600_1600_GridStep_20')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_5000_3000_400_400_GridStep_10')
data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800_GridStep20_depth')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_5000_3000_800_800_GridStep10_depth_train')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_3000_3000_1600_1600_GridStep20_depth_train')

# data_dir = os.path.join(data_sessions_outputs, 'berlin_many_angels_few_xys/-1520.15_1422.77/')
# data_dir = os.path.join(data_sessions_outputs, 'berlin_onlyPos_grid50/')
# data_dir = os.path.join(data_sessions_outputs, 'berlin_grid50/')
# data_dir = os.path.join(data_sessions_outputs, 'project_2017_09_06-12_40_19-grid_20/')
# data_dir = os.path.join(data_sessions_outputs, 'project_2017_09_06-21_41_07-grid_40/')
train_dir = os.path.join(data_dir, 'train')
# test_dir = None
test_dir = os.path.join(data_dir, 'test')

mesh_name = 'berlin'
grid_step = 20

roi = (4400, 5500, 800, 800)
# roi = (5000, 3000, 800, 800)
# roi = (3000, 4800, 1600, 1600)
# roi = (3000, 3000, 1600, 1600)
# roi = (5000, 3000, 400, 400)

# weights_filename = os.path.join(model_sessions_outputs,
#  'meshNet_2017_10_10-14_13_59-25Epochs-Grid20-almost-PoseNet/hdf5/meshNet_weights.e024-vloss0.3175.hdf5')
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_11_21-09_45_32/hdf5/meshNet_weights.e047-vloss0.5336.hdf5')
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_11_24-09_37_32_60Epochs_Berlin_Grid50/hdf5/meshNet_weights.e038-loss0.54771-vloss0.5626.hdf5')
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_11_28-13_53_12-100Epochs_Berlin_ROI_Grid200_NoAngles/hdf5/meshNet_weights.e093-loss0.07357-vloss0.5875.hdf5')

# Grid 200 - Single angle
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_06-12_40_01-100Epochs_Grid200_Batch4/hdf5/meshNet_weights.e098-loss0.07383-vloss0.4329.hdf5')

# Grid 400 - Single angle
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_06-13_24_29-100Epochs_Grid400_Batch8/hdf5/meshNet_weights.e085-loss0.04812-vloss0.1097.hdf5')
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_06-13_24_29-100Epochs_Grid400_Batch8/hdf5/meshNet_weights.e088-loss0.05448-vloss0.0942.hdf5')

# Angle only - Single XY
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_13-12_40_52-100Epochs_anglesOnly_Batch8/hdf5/meshNet_weights.e090-loss0.00323-vloss0.0023.hdf5')

# Grid 200 - XY+Angles - Upper 1/3
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_20-10_30_58/hdf5/meshNet_weights.e017-loss0.21648-vloss0.3123.hdf5')

# Grid 200- XY+Angles + Entire image + Edges - angles
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_31-08_53_29-41Epochs_berlinRoi_Grid200_FullImage_Edges/hdf5/' +
# 'meshNet_weights.e038-loss0.09036-vloss0.1261.hdf5')
# TODO: Run better EPOCH-
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_31-08_53_29_60Epochs_berlinRoi_Grid200_FullImage_Edges/hdf5/' +
#  'meshNet_weights.e055-loss0.07561-vloss0.1184.hdf5')

# Grid 200- XY+Angles + Entire image + Edges_and_faces - angle
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_27-00_21_50_55Epochs_berlinRoi_Grid200_FullImage_Edges_and_faces/hdf5/meshNet_weights.e026-loss0.08365-vloss0.1302.hdf5')

# Grid 200- XY+Angles + Entire image + Edges_and_faces - quaternions
# weights_filename = os.path.join(model_sessions_outputs,
# 'meshNet_2017_12_28-09_50_03_100Epochs_berlinRoi_Grid200_FullImage_Edges_and_faces_quternions/hdf5/meshNet_weights.e039-loss42.69920-vloss97.8118.hdf5')

# TODO: Run AGAIN (Weight already update - change data to correct ROI)-
# Step 20- XY+Angles + Entire image + Edges_and_faces - angle - ROI 5K_3K_800_800
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_01_08-22_10_56_55Epochs_berlinRoi_5K_3K_800_800_Grid_Step_20_FullImage_Edges_and_faces_angle',
#                                 'hdf5/meshNet_best_loss_weights.e049-loss0.04243-vloss0.1199.hdf5')

# Grid 200- XY+Angles + Entire image + Edges_and_faces - quaternions - Low quaternions loss 1/1/3 instead of 150/150/500
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_01_10-11_28_23_60Epochs_berlinRoi_Grid_Step_20_FullImage_Edges_and_faceas_quaternion-low_weight',
#                                 'hdf5/meshNet_best_loss_weights.e054-loss0.23080-vloss0.6674.hdf5')

# Grid Step 40- XY+Angles + Entire image + Edges_and_faces - quaternions - Low quaternions loss 0.5/0.5/1.5 instead of 150/150/500
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_01_17-17_58_46_16Epochs_berlinRoi_Grid_Step_40_FullImage_Edges_and_faces_quaternion-low_weight',
#                                 'hdf5/meshNet_best_loss_weights.e013-loss0.37574-vloss1.0122.hdf5')

# Grid Step 10- XY+Angles + Entire image + Edges_and_faces - quaternions - Low quaternions loss 0.5/0.5/1.5 instead of 150/150/500
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_01_15-18_24_58_27Epochs_berlinRoi_Grid_Step_10_FullImage_Edges_and_faces_quaternion-low_weight',
#                                 'hdf5/meshNet_best_loss_weights.e025-loss0.22022-vloss0.1463.hdf5')

# Grid Step 20- ROI 3000_4800_1600_1600 - XY+Angles + Entire image + Edges_and_faces - quaternions - Low quaternions loss 0.5/0.5/1.5 instead of 150/150/500
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_02_14-10_41_36_Train_120Epochs_berlinRoi_3000_4800_1600_1600_Grid_Step_20_FullImage_Edges_and_faces_quaternion-low_weight',
#                                 'hdf5/meshNet_best_loss_weights.e113-loss0.22775-vloss0.8356.hdf5')

# Grid Step 20- ROI 3000_3000_1600_1600 - XY+Angles + Entire image + Edges_and_faces - quaternions - Low quaternions loss 0.5/0.5/1.5 instead of 150/150/500
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_03_06-11_43_01_Train_120Epochs_berlinRoi_3000_3000_1600_1600_GridStep_20_FullImage_Edges_and_faces_quaternion-low_weight',
#                                 'hdf5/meshNet_best_loss_weights.e045-loss0.35340-vloss0.7885.hdf5')

# Grid Step 10- ROI 4400_5500_800_800 - XY+Angles + Entire image + stacked faces - quaternions - Low quaternions loss 0.5/0.5/1.5 instead of 150/150/500
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_03_16-14_37_14_Train_120Epochs_berlinRoi_4400_5500_800_800_GridStep10_stacked_faces/',
#                                 'hdf5/meshNet_best_loss_weights.e021-loss0.20523-vloss0.1398.hdf5')

# Grid Step 10- ROI 5000_3000_800_800 - XY+Angles + Entire image + stacked - quaternions - Low quaternions loss 0.5/0.5/1.5 instead of 150/150/500
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_03_18-12_04_25_Train_100Epochs_berlinRoi_5000_3000_800_800_GridStep10_stacked',
#                                 'hdf5/meshNet_best_loss_weights.e025-loss0.22284-vloss0.2038.hdf5')

# Grid Step 20- ROI 3000_3000_1600_1600 - XY+Angles + Entire image + stacked - quaternions - Low quaternions loss 0.5/0.5/1.5 instead of 150/150/500
# weights_filename = os.path.join(model_sessions_outputs,
#                                 'meshNet_2018_03_19-14_04_56_Train_120Epochs_berlinRoi_3000_3000_1600_1600_GridStep20_stacked',
#                                 'hdf5/meshNet_best_loss_weights.e035-loss0.39286-vloss0.4575.hdf5')

# Grid Step 20- ROI 4400 5500_800 800 - XY+Angles + Entire image + stacked - quaternions - Low quaternions loss 0.5/0.5/1.5 instead of 150/150/500
weights_filename = os.path.join(model_sessions_outputs,
                                'meshNet_2018_03_09-21_34_54_Train_200Epochs_berlinRoi_4400_5500_800_800_GridStep20_depth',
                                'hdf5/meshNet_best_loss_weights.e178-loss0.07943-vloss0.3533.hdf5')


# TODO: Can this be inferred in case we are just testing?
x_type = 'stacked_faces'       # 'edges', 'faces', 'gauss_blur_15', 'edges_on_faces', 'stacked_faces', 'depth'
y_type = 'quaternion'  # 'angle', 'quaternion', 'matrix'

use_cache = False
use_pickle = False
render_to_screen = False
evaluate = True
load_weights = False
initial_epoch = 0  # Should have the weights Epoch number + 1
mess = False

test_only = False
if test_only:
    load_weights = True
    evaluate = False

if initial_epoch != 0:
    load_weights = True

# Training options
batch_size = 32
save_best_only = True

debug_level = 0

if debug_level == 0:    # No Debug
    part_of_data = 1.0
    epochs = initial_epoch + 120
elif debug_level == 1:  # Medium Debug
    part_of_data = 3000
    epochs = 2
elif debug_level == 2:  # Full Debug
    part_of_data = 100
    epochs = 1
else:
    raise Exception("Invalid debug level " + str(debug_level))


# Script config
sess_info = utils.get_meshNet_session_info(mesh_name, model_type, roi, epochs, grid_step, test_only, load_weights,
                                           x_type, y_type, mess)


# TODO: Save some ~10 random sample images+labels to output dir - to make sure what the model trained on
def main():
    logger.Logger(sess_info)
    print("Entered %s" % sess_info.title)

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
                    part_of_data=part_of_data,
                    use_cache=use_cache)

    if mess:
        print("Creating a MESS - x_train shuffle")
        np.random.shuffle(loader.x_train)
        print("Minimal test size (100 samples) - Test will mean nothing")
        loader.x_test = loader.x_test[:100]
        loader.y_test = loader.y_test[:100]

    # visualize.show_data(loader.x_train, bg_color=(0, 255, 0))

    print("Getting model...")
    image_shape = loader.x_train.shape[1:]
    nb_outs = len(loader.y_train[0])

    if model_type == 'posenet':
        import posenet
        params = {'image_shape': image_shape, 'xy_nb_outs': 2, 'rot_nb_outs': nb_outs-2, 'multi_gpu': multi_gpu}
        model, model_name = posenet.posenet_train(**params)
    elif model_type == 'resnet':
        params = {'image_shape': image_shape, 'xy_nb_outs': 2, 'rot_nb_outs': nb_outs - 2, 'multi_gpu': multi_gpu}

        # import resnext
        # model, model_name = resnext.resnext_regression_train(**params)

        # import residual_network
        # model, model_name = residual_network.resnext_regression_train(**params)

        import resnet50
        model, model_name = resnet50.resnet50_regression_train(**params)

    elif model_type == 'fc':
        params = {'image_shape': image_shape, 'xy_nb_outs': 2, 'rot_nb_outs': nb_outs - 2, 'multi_gpu': multi_gpu}

        model, model_name = meshNet_model.meshNet_fc(**params)

    else:
        # Custom loss is mandatory to take 360 degrees into consideration
        # params = {'image_shape': image_shape, 'nb_outs': nb_outs, 'loss': meshNet_model.meshNet_loss}
        # model, model_name = meshNet_model.reg_2_conv_relu_mp_2_conv_relu_dense_dense(**params)
        # model, model_name = meshNet_model.reg_2_conv_relu_mp_2_conv_relu_dense_dense_bigger(**params)
        # model, model_name = meshNet_model.almost_VGG11_bn(**params)
        raise ValueError("Unsupported model type:", model_type)

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

    print("Model type:", model_type)
    print("Multi GPU:", multi_gpu)
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
        callbacks = meshNet_model.get_checkpoint(sess_info, is_classification=False, save_best_only=save_best_only,
                                                 tensor_board=False)

        if model_type == 'posenet':
            history = model.fit(loader.x_train, [loader.y_train[:, :2], loader.y_train[:, 2:],
                                                 loader.y_train[:, :2], loader.y_train[:, 2:],
                                                 loader.y_train[:, :2], loader.y_train[:, 2:]],
                                batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                                validation_data=(loader.x_test,
                                                 [loader.y_test[:, :2], loader.y_test[:, 2:],
                                                  loader.y_test[:, :2], loader.y_test[:, 2:],
                                                  loader.y_test[:, :2], loader.y_test[:, 2:]]),
                                shuffle=True, initial_epoch=initial_epoch, verbose=2)
        elif model_type == 'resnet':
            history = model.fit(loader.x_train, [loader.y_train[:, :2], loader.y_train[:, 2:]],
                                batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                                validation_data=(loader.x_test, [loader.y_test[:, :2], loader.y_test[:, 2:]]),
                                shuffle=True, initial_epoch=initial_epoch, verbose=2)
        elif model_type == 'fc':
            history = model.fit(loader.x_train, [loader.y_train[:, :2], loader.y_train[:, 2:]],
                                batch_size=batch_size, epochs=epochs, callbacks=callbacks,
                                validation_data=(loader.x_test, [loader.y_test[:, :2], loader.y_test[:, 2:]]),
                                shuffle=True, initial_epoch=initial_epoch, verbose=2)
        else:
            # history = model.fit(loader.x_train, loader.y_train, batch_size=batch_size, epochs=epochs,
            #                     callbacks=callbacks, validation_data=(loader.x_test, loader.y_test), shuffle=True,
            #                     initial_epoch=initial_epoch, verbose=2)
            raise ValueError("Unsupported model type:", model_type)

        weights_filename = meshNet_model.load_best_weights(model, sess_info)
    else:
        history = None

    if evaluate:
        print("Evaluating model. Test shape", loader.y_test.shape)
        if model_type == 'posenet':
            test_scores = model.evaluate(loader.x_test, [loader.y_test[:, :2], loader.y_test[:, 2:],
                                         loader.y_test[:, :2], loader.y_test[:, 2:],
                                         loader.y_test[:, :2], loader.y_test[:, 2:]], batch_size=batch_size, verbose=0)
        elif model_type == 'resnet':
            test_scores = model.evaluate(loader.x_test, [loader.y_test[:, :2], loader.y_test[:, 2:]],
                                         batch_size=batch_size, verbose=0)
        elif model_type == 'fc':
            test_scores = model.evaluate(loader.x_test, [loader.y_test[:, :2], loader.y_test[:, 2:]],
                                         batch_size=batch_size, verbose=0)
        else:
            # test_score = model.evaluate(loader.x_test, loader.y_test, batch_size=batch_size, verbose=0)
            raise ValueError("Unsupported model type:", model_type)

        print('Evaluate results:')
        for i, metric in enumerate(model.metrics_names):
            print(metric, ":", test_scores[i])

    # if not test_only:
    #     loader.save_pickle(sess_info)

    if model_type == 'posenet':
        # TODO: Choose best output according to test_score
        # detailed_evaluation(model, loader, 1)
        # detailed_evaluation(model, loader, 2)
        detailed_evaluation(model, loader, 3)
    elif model_type == 'resnet':
        detailed_evaluation(model, loader, 1)
    elif model_type == 'fc':
        detailed_evaluation(model, loader, 1)
    else:
        raise ValueError("Unsupported model type:", model_type)

    if history:
        visualize.visualize_history(history, sess_info, render_to_screen)

    print("Done")


def predict(model, x):
    # Predict each test sample and measure total time

    start_time = time.time()
    predictions = model.predict(x, batch_size=128, verbose=0)
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


def calc_stats(y_true, y_pred, normalized=False, dataset_name='dataset'):
    print("%s errors..." % dataset_name)
    xy_error = utils.xy_dist(y_true[:, :2], y_pred[:, :2])
    print("%s xy error. Mean %s, median %s, std %s" % (dataset_name, np.mean(xy_error), np.median(xy_error),
                                                       np.std(xy_error)))
    angle_error = utils.rotation_error(y_true[:, 2:], y_pred[:, 2:], normalized=normalized)
    print("%s angle error. Mean %s, median %s, std %s" % (dataset_name, np.mean(angle_error), np.median(angle_error),
                                                          np.std(angle_error)))

    return xy_error, angle_error


def inverse_transform(loader, normalized, y_train_pred, y_test_pred):
    if not normalized:
        y_train_true = loader.y_inverse_transform(loader.y_train)
        y_train_pred = loader.y_inverse_transform(y_train_pred)

        y_test_true = loader.y_inverse_transform(loader.y_test)
        y_test_pred = loader.y_inverse_transform(y_test_pred)
    else:
        y_train_true = loader.y_train
        y_test_true = loader.y_test

    return y_train_true, y_train_pred, y_test_true, y_test_pred


def errors_plot(xy_error_train, angle_error_train, xy_error_test, angle_error_test):
    plots_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir)

    max_xy_error = max(np.max(xy_error_test), np.max(xy_error_train))

    try:
        # Plot 2d heatmap histograms of the errors
        train_2d_hist_fname = os.path.join(plots_dir, sess_info.title + '_predictions_err_2d_hist.png')
        visualize.multiple_plots(1, 1, 2, 1)
        visualize.plot_2d_hist(xy_error_test, angle_error_test, False, (50, 50), title='Test Err 2D Histogram',
                               xlabel='XY err', ylabel='Angle err', xlim=[0, max_xy_error], ylim=[0, 180], show=False)
        visualize.multiple_plots(1, 1, 2, 2)
        visualize.plot_2d_hist(xy_error_train, angle_error_train, False, (50, 50), title='Train Err 2D Histogram',
                               xlabel='XY err', ylabel='Angle err', xlim=[0, max_xy_error], ylim=[0, 180],
                               show=render_to_screen, save_path=train_2d_hist_fname)

        # Plot 1D histograms of the errors
        xy_hist_fname = os.path.join(plots_dir, sess_info.title + '_predictions_xy_err_hist.png')
        visualize.multiple_plots(2, 1, 2, 1)
        visualize.plot_hist(xy_error_train, False, 50, title='Train XY err(%s-samples)' % len(xy_error_train),
                            ylabel='Samples', show=False)
        visualize.multiple_plots(2, 1, 2, 2)
        visualize.plot_hist(xy_error_test, False, 50, title='Test XY err(%s-samples)' % len(xy_error_test),
                            ylabel='Samples', show=render_to_screen, save_path=xy_hist_fname)

        angle_hist_fname = os.path.join(plots_dir, sess_info.title + '_predictions_angle_err_hist.png')
        visualize.multiple_plots(3, 1, 2, 1)
        visualize.plot_hist(angle_error_train, False, 50, title='Train angle err(%s-samples)' %
                            len(angle_error_train), ylabel='Samples', show=False)
        visualize.multiple_plots(3, 1, 2, 2)
        visualize.plot_hist(angle_error_test, False, 50, title='Test angle err(%s-samples)' %
                            len(angle_error_test), ylabel='Samples', show=render_to_screen, save_path=angle_hist_fname)
    except Exception as e:
        print("Warning: {}".format(e))


# visualize.view_prediction(data_dir, loader, y_train_pred, xy_error_train, idx=5)
def detailed_evaluation(model, loader, output_number):
    """ output_number: PoseNet has total of 6 outputs. 3 pairs of [xy_out, rot_out]
                       ResNet/FC has single output pair [xy_out, rot_out]
    """
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

    print("Using output number:", output_number)
    xyz_output = (output_number - 1) * 2
    rotation_output = xyz_output + 1
    y_train_pred = np.concatenate((y_train_pred[xyz_output], y_train_pred[rotation_output]), axis=-1)
    y_test_pred = np.concatenate((y_test_pred[xyz_output], y_test_pred[rotation_output]), axis=-1)

    normalized = False
    print("Normalized:", normalized)
    y_train_true, y_train_pred, y_test_true, y_test_pred = inverse_transform(loader, normalized, y_train_pred, y_test_pred)

    xy_error_train, angle_error_train = calc_stats(y_train_true, y_train_pred, normalized=normalized,
                                                   dataset_name='Train')
    xy_error_test, angle_error_test = calc_stats(y_test_true, y_test_pred, normalized=normalized, dataset_name='Test')

    # for i in [0, 1, 2, 1000, 4000, len(y_train_pred)-1]:
    #     visualize.view_prediction(data_dir, roi, loader, y_train_pred, y_test_pred, errors_by='xy', idx=i, is_train=True,
    #                               normalized=False, asc=False, figure_num=i)
    # for i in [0, 1, 2, 1000, 4000, len(y_train_pred)-1]:
    #     visualize.view_prediction(data_dir, roi, loader, y_train_pred, y_test_pred, errors_by='angle', idx=i, is_train=True,
    #                               normalized=False, asc=False, figure_num=i)
    # for i in [0, 1, 2, 1000, 4000, len(y_train_pred)-1]:
    #     visualize.view_prediction(data_dir, roi, loader, y_train_pred, y_test_pred, errors_by='comb', idx=i, is_train=True,
    #                               normalized=False, asc=False, figure_num=i)

    data = {
        "model_type": model_type,
        "weights_filename": weights_filename,
        "x_type": x_type,
        "y_type": y_type,
        "train_dir": train_dir,
        "test_dir": test_dir,
        "image_size": loader.image_size,

        "test_only": test_only,
        "mesh_name": mesh_name,
        "roi": roi,
        "grid_step": grid_step,
        "epochs": epochs,
        "batch_size": batch_size,
        "part_of_data": part_of_data,

        "file_urls_train": loader.file_urls_train,
        "y_train_true": y_train_true,
        "y_train_pred": y_train_pred,
        "xy_error_train": xy_error_train,
        "angle_error_train": angle_error_train,

        "file_urls_test": loader.file_urls_test,
        "y_test_true": y_test_true,
        "y_test_pred": y_test_pred,
        "xy_error_test": xy_error_test,
        "angle_error_test": angle_error_test,
    }
    utils.save_pickle(sess_info, data)

    errors_plot(xy_error_train, angle_error_train, xy_error_test, angle_error_test)


if __name__ == '__main__':
    main()
