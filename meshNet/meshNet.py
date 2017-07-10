from __future__ import print_function

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

weights_filename = ""

test_only = False
if test_only:
    load_weights = True
    load_pickle = True
    save_pickle = False
else:
    load_weights = False
    load_pickle = False
    save_pickle = True

# Training options
batch_size = 16

debug_level = 0

if debug_level == 0:    # No Debug
    part_of_data = 1.0
    nb_epoch = 5
elif debug_level == 1:   # Medium Debug
    part_of_data = 0.1
    nb_epoch = 5
elif debug_level == 2:  # Full Debug
    part_of_data = 0.005
    nb_epoch = 2
else:
    raise Exception("Invalid debug level " + str(debug_level))


def main():
    logger.Logger(sess_info)
    print("Entered %s" % title)

    x_train, x_test, \
        y_train, y_test, \
        file_urls_train, file_urls_test = meshNet_loader.load_data(
            '/home/moti/cg/project/sample_images/', part_of_data,
            pkl_file_path='/home/moti/cg/project/meshNet/sessions_outputs/mesh_data.pkl', image_size=(100, 75))


    # visualize.show_data(x_train)

    # x_train, _, x_test, labels_train, _, labels_test, y_train, _, y_test, encoder, scaler, image_shape, classes_num = \
    #     preprocessing.preprocess(x_train, None, x_test, labels_train, None, labels_test, encoder_params=encoder_params,
    #                              augmentation_params=augmentation_params, scaler_params=scaler_params,
    #                              dim_ordering=dim_ordering)

    print("Getting model...")
    image_shape = x_train.shape[1:]
    nb_outs = len(y_train[0])
    # print("image_shape: %d " % image_shape)
    # print("nb_outs: %d " % nb_outs)

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
        callbacks = meshNet_model.get_checkpoint(sess_info, False, tensor_board=False)

        # class_weight = {0: 20.,
        #                 1: 20.,
        #                 2: 1.,
        #                 3: 1.}
        history = model.fit(x_train, y_train, batch_size=batch_size, epochs=nb_epoch, callbacks=callbacks,
                            validation_data=(x_test, y_test), shuffle=True)  # , class_weight=class_weight)
    else:
        history = None

    test_score = model.evaluate(x_test, y_test, verbose=0)
    print('Test score:', test_score)

    if save_pickle:
        utils.save_pickle(sess_info, file_urls_train, y_train, None, None, file_urls_test, y_test)

    if history:
        visualize.visualize_history(history, sess_info)

if __name__ == '__main__':
    main()
