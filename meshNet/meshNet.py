from __future__ import print_function
from __future__ import division

import time
import os

model_type = 'resnet'  # 'posenet'/'resnet'/'fc'

multi_gpu = False
if not multi_gpu:
    os.environ["CUDA_VISIBLE_DEVICES"] = "1"
    # os.environ["CUDA_VISIBLE_DEVICES"] = ""  # CPU only

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
    project_base_dir = '/home/moti/cg/project'
else:
    # project_base_dir = '/mnt/SSD1/moti/project'
    project_base_dir = '/mnt/arik_2T_usb/project'

data_sessions_outputs = os.path.join(project_base_dir, 'sessions_outputs')
model_sessions_outputs = os.path.join(project_base_dir, 'meshNet/sessions_outputs')

# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200/')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid400/')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid800/')
# data_dir = os.path.join(data_sessions_outputs, 'berlin_angleOnly_4950_5850/')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200_Full/')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800Grid200_FullImage')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_5000_3000_800_800_GridStep_20')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800_GridStep_40')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800_GridStep_10')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_3000_4800_1600_1600_GridStep_20')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_3000_3000_1600_1600_GridStep_20')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_5000_3000_400_400_GridStep_10')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_4400_5500_800_800_GridStep20_depth')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_5000_3000_800_800_GridStep10_depth')
# data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_3000_3000_1600_1600_GridStep20_depth')

# data_dir = os.path.join(data_sessions_outputs, 'berlin_many_angels_few_xys/-1520.15_1422.77/')
# data_dir = os.path.join(data_sessions_outputs, 'berlin_onlyPos_grid50/')
# data_dir = os.path.join(data_sessions_outputs, 'berlin_grid50/')
# data_dir = os.path.join(data_sessions_outputs, 'project_2017_09_06-12_40_19-grid_20/')
# data_dir = os.path.join(data_sessions_outputs, 'project_2017_09_06-21_41_07-grid_40/')

data_dir = os.path.join(data_sessions_outputs, 'berlinRoi_-1600_-800_1600_1600_GridStep10')

train_dir = os.path.join(data_dir, 'train')
# test_dir = None
test_dir = os.path.join(data_dir, 'test')

mesh_name = 'berlin'

# roi = (-1600, -800, 400, 400)
# grid_step = 20
roi = None
grid_step = None

# roi = (4400, 5500, 800, 800)
# roi = (5000, 3000, 800, 800)
# roi = (3000, 4800, 1600, 1600)
# roi = (3000, 3000, 1600, 1600)
# roi = (5000, 3000, 400, 400)

weights_filename = None

# TODO: Can this be inferred in case we are just testing?
x_type = None  # 'edges', 'faces', 'gauss_blur_15', 'edges_on_faces', 'stacked_faces', 'depth'
y_type = 'quaternion'  # 'angle', 'quaternion', 'matrix'

use_cache = True
render_to_screen = False
evaluate = True
load_weights = False
initial_epoch = 0  # Should have the weights Epoch number + 1
fine_tune = False
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

sess_info = None


def main():
    global weights_filename

    print("Entered %s" % sess_info.title)

    loader = meshNet_loader.DataLoader()
    loader.load(train_dir=train_dir,
                test_dir=test_dir,
                x_range=(0, 1),
                x_type=x_type,
                y_type=y_type,
                part_of_data=part_of_data,
                use_cache=use_cache,
                auto_collect=True, roi=roi, grid_step=grid_step)

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

    if fine_tune:
        if not load_weights:  # Sanity
            raise Exception("Something is wrong. Freezing untrained weights does not make much sense")

        if model_type == 'resnet':
            print("Freeze the layers except the last 4 layers...")
            for layer in model.layers[:-14]:
                layer.trainable = False

            model = resnet50.model_compile(model, multi_gpu)
        else:
            raise ValueError("Unsupported model type for fine_tuning:", model_type)

    print("Check the trainable status of the individual layers")
    for counter, layer in enumerate(model.layers):
        print(counter, layer, "isTrainable:", layer.trainable)
    print("")

    print("image_shape:", image_shape)
    print("nb_outs:", nb_outs)
    print("Train shape:", loader.x_train.shape)
    print("Test shape:", loader.x_test.shape)
    print("")

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

    else:
        history = None

    hdf5_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, 'hdf5')
    if not test_only:
        weights_list = os.listdir(hdf5_dir)
        weights_list.sort()
    else:
        weights_list = ["dummy"]

    for weights_fname in weights_list:
        if not test_only:
            weights_filename = os.path.join(hdf5_dir, weights_fname)
            meshNet_model.load_model_weights(model, weights_filename)

        if evaluate and not mess:
            print("Evaluating model. Test shape", loader.y_test.shape)
            if model_type == 'posenet':
                test_scores = model.evaluate(loader.x_test, [loader.y_test[:, :2], loader.y_test[:, 2:],
                                             loader.y_test[:, :2], loader.y_test[:, 2:],
                                             loader.y_test[:, :2], loader.y_test[:, 2:]], batch_size=batch_size,
                                             verbose=0)
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

        try:
            if model_type == 'posenet':
                # detailed_evaluation(model, loader, 1)
                # detailed_evaluation(model, loader, 2)
                detailed_evaluation(model, loader, 3)
            elif model_type == 'resnet':
                detailed_evaluation(model, loader, 1)
            elif model_type == 'fc':
                detailed_evaluation(model, loader, 1)
            else:
                raise ValueError("Unsupported model type:", model_type)
        except Exception as e:
            print("detailed evaluation failed. Warning: {}".format(e))

    if history is not None:
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

    print("xy (mean, median, std), angle (mean, median, std): (%s, %s, %s, %s, %s, %s)" %
          (np.mean(xy_error), np.median(xy_error), np.std(xy_error),
           np.mean(angle_error), np.median(angle_error), np.std(angle_error)))

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
    print("errors_plot: Entered")
    plots_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir)

    max_xy_error = max(np.max(xy_error_test), np.max(xy_error_train))

    try:
        print("errors_plot: Plot 2d heatmap histograms of the errors")
        train_2d_hist_fname = os.path.join(plots_dir, sess_info.title + '_predictions_err_2d_hist.png')
        visualize.multiple_plots(1, 1, 2, 1)
        visualize.plot_2d_hist(xy_error_test, angle_error_test, False, (50, 50), title='Test Err 2D Histogram',
                               xlabel='XY err', ylabel='Angle err', xlim=[0, max_xy_error], ylim=[0, 180], show=False)
        visualize.multiple_plots(1, 1, 2, 2)
        visualize.plot_2d_hist(xy_error_train, angle_error_train, False, (50, 50), title='Train Err 2D Histogram',
                               xlabel='XY err', ylabel='Angle err', xlim=[0, max_xy_error], ylim=[0, 180],
                               show=render_to_screen, save_path=train_2d_hist_fname)

        print("errors_plot: Plot 1D histograms of the xy errors")
        xy_hist_fname = os.path.join(plots_dir, sess_info.title + '_predictions_xy_err_hist.png')
        visualize.multiple_plots(2, 1, 2, 1)
        visualize.plot_hist(xy_error_train, False, 50, title='Train XY err(%s-samples)' % len(xy_error_train),
                            ylabel='Samples', show=False)
        visualize.multiple_plots(2, 1, 2, 2)
        visualize.plot_hist(xy_error_test, False, 50, title='Test XY err(%s-samples)' % len(xy_error_test),
                            ylabel='Samples', show=render_to_screen, save_path=xy_hist_fname)

        print("errors_plot: Plot 1D histograms of the angle errors")
        angle_hist_fname = os.path.join(plots_dir, sess_info.title + '_predictions_angle_err_hist.png')
        visualize.multiple_plots(3, 1, 2, 1)
        visualize.plot_hist(angle_error_train, False, 50, title='Train angle err(%s-samples)' %
                            len(angle_error_train), ylabel='Samples', show=False)
        visualize.multiple_plots(3, 1, 2, 2)
        visualize.plot_hist(angle_error_test, False, 50, title='Test angle err(%s-samples)' %
                            len(angle_error_test), ylabel='Samples', show=render_to_screen, save_path=angle_hist_fname)
    except Exception as e:
        print("Warning: {}".format(e))
    print("errors_plot: Done")


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
    y_train_true, y_train_pred, y_test_true, y_test_pred = inverse_transform(loader, normalized, y_train_pred,
                                                                             y_test_pred)

    xy_error_train, angle_error_train = calc_stats(y_train_true, y_train_pred, normalized=normalized,
                                                   dataset_name='Train')
    xy_error_test, angle_error_test = calc_stats(y_test_true, y_test_pred, normalized=normalized, dataset_name='Test')

    # for i in [0, 1, 2, 1000, 4000, len(y_train_pred)-1]:
    #     visualize.view_prediction(data_dir, roi, loader, y_train_pred, y_test_pred, errors_by='xy', idx=i,
    #                                is_train=True, normalized=False, asc=False, figure_num=i)
    # for i in [0, 1, 2, 1000, 4000, len(y_train_pred)-1]:
    #     visualize.view_prediction(data_dir, roi, loader, y_train_pred, y_test_pred, errors_by='angle', idx=i,
    #                                is_train=True, normalized=False, asc=False, figure_num=i)
    # for i in [0, 1, 2, 1000, 4000, len(y_train_pred)-1]:
    #     visualize.view_prediction(data_dir, roi, loader, y_train_pred, y_test_pred, errors_by='comb', idx=i,
    #                                is_train=True, normalized=False, asc=False, figure_num=i)

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

    pickle_title = os.path.basename(os.path.splitext(weights_filename)[0])
    utils.save_pickle(sess_info, data, pickle_title)

    print("Handling TRAIN")
    k = 5
    utils.train_result(y_train_true, y_train_pred, k)

    print("Handling TEST")
    xy_step = data['grid_step']
    utils.test_result(y_test_true, y_test_pred, xy_step, k=100)

    errors_plot(xy_error_train, angle_error_train, xy_error_test, angle_error_test)


if __name__ == '__main__':
    session_list = [
        # (roi, grid_step, x_type, load_weights, initial_epoch, weights_filename, fine_tune)
        # debug
        # ((-1300, -800, 50, 50), 20, 'edges', False, 0, None, False),  # debug
        # ((-1300, -800, 50, 50), 20, 'edges', False, 0, None, False),  # debug

        # 400x400
        # ((-1600, -800, 400, 400), 20, 'edges', False, 0, None, False),
        # ((-1600, -800, 400, 400), 20, 'stacked_faces', False, 0, None, False),
        # ((-1600, -800, 400, 400), 20, 'depth', False, 0, None, False),

        # ((-1200, -800, 400, 400), 20, 'edges', False, 0, None, False),
        # ((-1200, -800, 400, 400), 20, 'stacked_faces', False, 0, None, False),
        # ((-1200, -800, 400, 400), 20, 'depth', False, 0, None, False),

        # ((-1600, -400, 400, 400), 20, 'edges', False, 0, None, False),
        # ((-1600, -400, 400, 400), 20, 'stacked_faces', False, 0, None, False),
        # ((-1600, -400, 400, 400), 20, 'depth', False, 0, None, False),

        # ((-1200, -400, 400, 400), 20, 'edges', False, 0, None, False),
        # ((-1200, -400, 400, 400), 20, 'stacked_faces', False, 0, None, False),
        # ((-1200, -400, 400, 400), 20, 'depth', False, 0, None, False),

        # 800x800
        # ((-1600, -800, 800, 800), 20, 'edges', False, 0, None, False),
        # ((-1600, -800, 800, 800), 20, 'stacked_faces', False, 0, None, False),
        # ((-1600, -800, 800, 800), 20, 'depth', False, 0, None, False),

        # Resume training
        # ((-1600, -800, 800, 800), 20, 'edges', True, 120, os.path.join(model_sessions_outputs, 'gridStep20/-1600_-800_800_800/meshNet_2018_06_05-06_50_53_Train_resnet_120Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_edges/hdf5/meshNet_best_loss_weights.e119-loss0.01850-vloss0.2434.hdf5'), False),
        # Run with test_only - test on best loss
        # ((-1600, -800, 800, 800), 20, 'edges', True, 214, os.path.join(model_sessions_outputs, 'meshNet_2018_11_12-16_41_51_TrainResume_resnet_360Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_edges/hdf5/meshNet_best_loss_weights.e213-loss0.01469-vloss0.2351.hdf5'), False),
        # Run with test_only - test on best val
        # ((-1600, -800, 800, 800), 20, 'edges', True, 197, os.path.join(model_sessions_outputs, 'meshNet_2018_11_12-16_41_51_TrainResume_resnet_360Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_edges/hdf5/meshNet_best_val_loss_weights.e196-loss0.01507-vloss0.2347.hdf5'), False),

        # ((-1600, -800, 800, 800), 20, 'stacked_faces', True, 120, os.path.join(model_sessions_outputs, 'gridStep20/-1600_-800_800_800/meshNet_2018_06_06-23_00_47_Train_resnet_120Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_stacked_faces/hdf5/meshNet_best_loss_weights.e119-loss0.01717-vloss0.1720.hdf5'), False),

        # Try train on (-1600, -800, 800, 800) weights. Does it speed up convergence?
        # ((-800, -800, 800, 800), 20, 'edges', True, 0, os.path.join(model_sessions_outputs, 'gridStep20/-1600_-800_800_800/meshNet_2018_06_05-06_50_53_Train_resnet_120Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_edges/hdf5/meshNet_best_loss_weights.e119-loss0.01850-vloss0.2434.hdf5'), False),
        # ((-800, -800, 800, 800), 20, 'stacked_faces', True, 0, os.path.join(model_sessions_outputs, 'gridStep20/-1600_-800_800_800/meshNet_2018_06_06-23_00_47_Train_resnet_120Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_stacked_faces/hdf5/meshNet_best_loss_weights.e119-loss0.01717-vloss0.1720.hdf5'), False),
        # ((-800, -800, 800, 800), 20, 'depth', True, 0, os.path.join(model_sessions_outputs, 'gridStep20/-1600_-800_800_800/meshNet_2018_06_08-14_22_25_Train_resnet_240Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_depth/hdf5/meshNet_best_loss_weights.e238-loss0.01335-vloss0.1813.hdf5'), False),

        # Run area again to compare result when trained without pre-train weights - Should be used for graph in the paper
        # ((-800, -800, 800, 800), 20, 'stacked_faces', False, 0, None, False),
        # Run area again to compare result when trained with pre-train weights of another area and grid step - consider using in the paper
        # ((-800, -800, 800, 800), 20, 'stacked_faces', True, 0, os.path.join(model_sessions_outputs, 'gridStep10/-1600_-800_400_400/meshNet_2018_07_12-01_02_18_Train_resnet_240Epochs_berlin_ROI_-1600_-800_400_400_GridStep10_quaternion_stacked_faces/hdf5/meshNet_best_loss_weights.e239-loss0.01256-vloss0.0403.hdf5'), False),

        ((-800, -800, 800, 800), 20, 'edges', False, 0, None, False),
        ((-800, -800, 800, 800), 20, 'stacked_faces', False, 0, None, False),
        ((-800, -800, 800, 800), 20, 'depth', False, 0, None, False),

        # ((-1600, 0, 800, 800), 20, 'edges', False, 0, None, False),
        # ((-1600, 0, 800, 800), 20, 'stacked_faces', False, 0, None, False),
        # ((-1600, 0, 800, 800), 20, 'depth', False, 0, None, False),

        # 1600x1600
        # ((-1600, -800, 1600, 1600), 20, 'edges', True, 161, os.path.join(model_sessions_outputs, 'meshNet_2018_10_18-11_03_39_Train_resnet_240Epochs_berlin_ROI_-1600_-800_1600_1600_GridStep20_quaternion_edges/hdf5/meshNet_best_loss_weights.e160-loss0.01573-vloss0.2331.hdf5'), False),
        # DID not run - MemoryError ((-1600, -800, 1600, 1600), 20, 'stacked_faces', False, 0, None, False),
        # DID not run - MemoryError ((-1600, -800, 1600, 1600), 20, 'depth', False, 0, None, False),

        # Resume 400x400
        #((-1600, -800, 400, 400), 20, 'edges', True, 120, os.path.join(model_sessions_outputs, 'meshNet_2018_05_31-22_11_07_Train_resnet_120Epochs_berlin_ROI_-1600_-800_400_400_GridStep20_quaternion_edges', 'hdf5', 'meshNet_best_loss_weights.e119-loss0.02682-vloss0.2378.hdf5'), False),
        #((-1600, -800, 400, 400), 20, 'stacked_faces', True, 119, os.path.join(model_sessions_outputs, 'meshNet_2018_06_01-05_41_09_Train_resnet_120Epochs_berlin_ROI_-1600_-800_400_400_GridStep20_quaternion_stacked_faces', 'hdf5', 'meshNet_best_loss_weights.e118-loss0.02585-vloss0.1679.hdf5'), False),
        #((-1600, -800, 400, 400), 20, 'depth', True, 117, os.path.join(model_sessions_outputs, 'meshNet_2018_06_01-12_25_00_Train_resnet_120Epochs_berlin_ROI_-1600_-800_400_400_GridStep20_quaternion_depth', 'hdf5', 'meshNet_best_loss_weights.e116-loss0.02618-vloss0.1688.hdf5'), False),

        #((-1200, -800, 400, 400), 20, 'edges', True, 120, os.path.join(model_sessions_outputs, 'meshNet_2018_06_01-18_27_29_Train_resnet_120Epochs_berlin_ROI_-1200_-800_400_400_GridStep20_quaternion_edges', 'hdf5', 'meshNet_best_val_loss_weights.e119-loss0.02469-vloss0.2412.hdf5'), False),
        #((-1200, -800, 400, 400), 20, 'stacked_faces', True, 119, os.path.join(model_sessions_outputs, 'meshNet_2018_06_02-02_14_48_Train_resnet_120Epochs_berlin_ROI_-1200_-800_400_400_GridStep20_quaternion_stacked_faces', 'hdf5', 'meshNet_best_loss_weights.e118-loss0.02513-vloss0.2074.hdf5'), False),
        #((-1200, -800, 400, 400), 20, 'depth', True, 120, os.path.join(model_sessions_outputs, 'meshNet_2018_06_02-09_54_40_Train_resnet_120Epochs_berlin_ROI_-1200_-800_400_400_GridStep20_quaternion_depth', 'hdf5', 'meshNet_best_loss_weights.e119-loss0.02408-vloss0.2183.hdf5'), False),

        # 400x400 - Step 10
        # ((-1600, -800, 400, 400), 10, 'edges', False, 0, None, False),
        # ((-1600, -800, 400, 400), 10, 'stacked_faces', False, 0, None, False),
        # ((-1600, -800, 400, 400), 10, 'depth', False, 0, None, False),

        # ((-1200, -800, 400, 400), 10, 'edges', True, 0, os.path.join(model_sessions_outputs, 'gridStep10/-1600_-800_400_400/meshNet_2018_07_10-16_12_33_Train_resnet_240Epochs_berlin_ROI_-1600_-800_400_400_GridStep10_quaternion_edges/hdf5/meshNet_best_loss_weights.e238-loss0.01251-vloss0.0665.hdf5'), False),
        # ((-1200, -800, 400, 400), 10, 'stacked_faces', True, 0, os.path.join(model_sessions_outputs, 'gridStep10/-1600_-800_400_400/meshNet_2018_07_12-01_02_18_Train_resnet_240Epochs_berlin_ROI_-1600_-800_400_400_GridStep10_quaternion_stacked_faces/hdf5/meshNet_best_loss_weights.e239-loss0.01256-vloss0.0403.hdf5'), False),
        # ((-1200, -800, 400, 400), 10, 'depth', True, 0, os.path.join(model_sessions_outputs, 'gridStep10/-1600_-800_400_400/meshNet_2018_07_13-09_37_01_Train_resnet_240Epochs_berlin_ROI_-1600_-800_400_400_GridStep10_quaternion_depth/hdf5/meshNet_best_loss_weights.e236-loss0.01246-vloss0.0487.hdf5'), False),

        # ((-1600, -400, 400, 400), 10, 'edges', False, 0, None, False),
        # ((-1600, -400, 400, 400), 10, 'stacked_faces', False, 0, None, False),
        # ((-1600, -400, 400, 400), 10, 'depth', False, 0, None, False),

        # ((-1200, -400, 400, 400), 10, 'edges', False, 0, None, False),
        # ((-1200, -400, 400, 400), 10, 'stacked_faces', False, 0, None, False),
        # ((-1200, -400, 400, 400), 10, 'depth', False, 0, None, False),

        # Done # ((-1200, -400, 800, 800), 10, 'edges', False, 0, None, False),
        # Memory Error # ((-1200, -400, 800, 800), 10, 'stacked_faces', False, 0, None, False),
        # ((-1200, -400, 800, 800), 10, 'depth', False, 0, None, False),

        # Step 40
        # ((-1600, -800, 1600, 1600), 40, 'edges', False, 0, None, False),
        # ((-1600, -800, 1600, 1600), 40, 'stacked_faces', False, 0, None, False),
        # ((-1600, -800, 1600, 1600), 40, 'depth', False, 0, None, False),

        # ((-800, 0, 800, 800), 40, 'edges', False, 0, None, False),
        # ((-800, 0, 800, 800), 40, 'stacked_faces', False, 0, None, False),
        # ((-800, 0, 800, 800), 40, 'depth', False, 0, None, False),

        # Fine Tune
        # ((-1300, -800, 50, 50), 20, 'edges', True, 0, os.path.join(model_sessions_outputs, 'debug_meshNet_Train_resnet_berlin_ROI_-1300_-800_50_50_GridStep20_quaternion_edges/hdf5/meshNet_best_loss_weights.e004-loss1.21845-vloss4.7761.hdf5'), True),  # debug

        # ((-800, -800, 800, 800), 20, 'edges', True, 0, os.path.join(model_sessions_outputs, 'gridStep20/-1600_-800_800_800/meshNet_2018_06_05-06_50_53_Train_resnet_120Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_edges/hdf5/meshNet_best_loss_weights.e119-loss0.01850-vloss0.2434.hdf5'), True),
        # ((-800, -800, 800, 800), 20, 'stacked_faces', True, 0, os.path.join(model_sessions_outputs, 'gridStep20/-1600_-800_800_800/meshNet_2018_06_06-23_00_47_Train_resnet_120Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_stacked_faces/hdf5/meshNet_best_loss_weights.e119-loss0.01717-vloss0.1720.hdf5'), True),
        # ((-800, -800, 800, 800), 20, 'depth', True, 0, os.path.join(model_sessions_outputs, 'gridStep20/-1600_-800_800_800/meshNet_2018_06_08-14_22_25_Train_resnet_240Epochs_berlin_ROI_-1600_-800_800_800_GridStep20_quaternion_depth/hdf5/meshNet_best_loss_weights.e238-loss0.01335-vloss0.1813.hdf5'), True),

        # ((-1200, -800, 400, 400), 10, 'edges', True, 0, os.path.join(model_sessions_outputs, 'gridStep10/-1600_-800_400_400/meshNet_2018_07_10-16_12_33_Train_resnet_240Epochs_berlin_ROI_-1600_-800_400_400_GridStep10_quaternion_edges/hdf5/meshNet_best_loss_weights.e238-loss0.01251-vloss0.0665.hdf5'), True),
        # ((-1200, -800, 400, 400), 10, 'stacked_faces', True, 0, os.path.join(model_sessions_outputs, 'gridStep10/-1600_-800_400_400/meshNet_2018_07_12-01_02_18_Train_resnet_240Epochs_berlin_ROI_-1600_-800_400_400_GridStep10_quaternion_stacked_faces/hdf5/meshNet_best_loss_weights.e239-loss0.01256-vloss0.0403.hdf5'), True),
        # ((-1200, -800, 400, 400), 10, 'depth', True, 0, os.path.join(model_sessions_outputs, 'gridStep10/-1600_-800_400_400/meshNet_2018_07_13-09_37_01_Train_resnet_240Epochs_berlin_ROI_-1600_-800_400_400_GridStep10_quaternion_depth/hdf5/meshNet_best_loss_weights.e236-loss0.01246-vloss0.0487.hdf5'), True),
    ]

    idx = 0
    for roi, grid_step, x_type, load_weights, initial_epoch, weights_filename, fine_tune in session_list:
        epochs = initial_epoch + 240

        sess_info = utils.get_meshNet_session_info(mesh_name, model_type, roi, epochs, grid_step, test_only,
                                                   load_weights, x_type, y_type, mess, fine_tune)
        log = logger.Logger(sess_info)

        print("")
        print("New session [%s]: roi [%s], grid_step [%s], x_type [%s]" % (idx, roi, grid_step, x_type))
        main()
        print("Done session [%s]: roi [%s], grid_step [%s], x_type [%s]" % (idx, roi, grid_step, x_type))
        print("")
        idx += 1

        log.close()
