import os

import numpy as np
from keras.engine import Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D, Lambda
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model
from keras.optimizers import SGD, adadelta, Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import ModelCheckpoint, TensorBoard

from keras import backend as K

from keras import initializers as keras_initializers
from keras import backend as keras_backend

import utils
import consts


def constant_init(value=0.1):
    return keras_initializers.Constant(value=value)


def load_model_weights(model, weights_full_path):
    print("Loading weights: " + weights_full_path)
    model.load_weights(weights_full_path)


def get_checkpoint(sess_info, isClassification, tensor_board=False):
    hdf5_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, 'hdf5')
    utils.mkdirs(hdf5_dir)

    if keras_backend.backend() != 'tensorflow':
        tensor_board = False

    tensor_board_cp = None
    if tensor_board:
        tensor_board_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, 'tensor_board')
        utils.mkdirs(tensor_board_dir)
        tensor_board_cp = TensorBoard(log_dir=tensor_board_dir, histogram_freq=1, write_graph=True)

    if isClassification:
        monitor = "val_loss"
        extra_params = "_vacc{val_acc:.3f}"
    else:
        monitor = "val_loss"
        extra_params = ""

    hdf5_fname = sess_info.title + "_weights.e{epoch:03d}-vloss{val_loss:.4f}" + extra_params + ".hdf5"
    hdf5_full_path = os.path.join(hdf5_dir, hdf5_fname)

    model_cp = ModelCheckpoint(filepath=hdf5_full_path, verbose=0, save_best_only=True, monitor=monitor)
    if tensor_board:
        return [model_cp, tensor_board_cp]
    else:
        return [model_cp]


# FIXME: Add paper reference
def reg_2_conv_relu_mp_2_conv_relu_dense_dense(image_shape, nb_outs, optimizer=None, loss=None):
    model_name = reg_2_conv_relu_mp_2_conv_relu_dense_dense.__name__

    model = Sequential()

    model.add(Conv2D(16, (5, 5), padding='same', input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(120, (3, 3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(84))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Dense(nb_outs))

    if optimizer is None:
        optimizer = RMSprop()
    if loss is None:
        loss = "mean_squared_error"

    model.compile(loss=loss, optimizer=optimizer)

    return model, model_name