import os
import warnings

import numpy as np
from keras.engine import Input
from keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, SeparableConv2D, Lambda, ZeroPadding2D, UpSampling2D
from keras.layers import Dense, Dropout, Activation, Flatten, BatchNormalization, merge, Reshape, Permute, Layer
from keras.layers.advanced_activations import PReLU
from keras.models import Sequential, Model
from keras.optimizers import SGD, Adadelta, Adam, RMSprop
from keras.regularizers import l2
from keras.callbacks import Callback, ModelCheckpoint, TensorBoard

from keras import backend as K

from keras import initializers as keras_initializers

from googlenet_custom_layers import PoolHelper, LRN

import utils
import consts


def constant_init(value=0.1):
    return keras_initializers.Constant(value=value)


def load_model_weights(model, weights_full_path):
    print("Loading weights: " + weights_full_path)
    model.load_weights(weights_full_path)


# Assumes best model is last saved model (save_best_model parameter of ModelCheckpoint)
def load_best_weights(model, sess_info):
    hdf5_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, 'hdf5')
    weights_list = os.listdir(hdf5_dir)
    weights_list.sort(reverse=True)

    weights_full_path = os.path.join(hdf5_dir, weights_list[0])
    print("Loading best weights: " + weights_full_path)
    load_model_weights(model, weights_full_path)


# Some modifications to Keras ModelCheckpoint, currently mainly to save disk space
class myModelCheckpoint(Callback):
    """Save the model after every epoch.

    `filepath` can contain named formatting options,
    which will be filled the value of `epoch` and
    keys in `logs` (passed in `on_epoch_end`).

    For example: if `filepath` is `weights.{epoch:02d}-{val_loss:.2f}.hdf5`,
    then the model checkpoints will be saved with the epoch number and
    the validation loss in the filename.

    # Arguments
        filepath: string, path to save the model file.
        monitor: quantity to monitor.
        verbose: verbosity mode, 0 or 1.
        save_best_only: if `save_best_only=True`,
            the latest best model according to
            the quantity monitored will not be overwritten.
        mode: one of {auto, min, max}.
            If `save_best_only=True`, the decision
            to overwrite the current save file is made
            based on either the maximization or the
            minimization of the monitored quantity. For `val_acc`,
            this should be `max`, for `val_loss` this should
            be `min`, etc. In `auto` mode, the direction is
            automatically inferred from the name of the monitored quantity.
        save_weights_only: if True, then only the model's weights will be
            saved (`model.save_weights(filepath)`), else the full model
            is saved (`model.save(filepath)`).
        period: Interval (number of epochs) between checkpoints.
    """

    def __init__(self, filepath, monitor='val_loss', verbose=0,
                 save_best_only=False, save_weights_only=False,
                 mode='auto', period=1, remove_last_best=True):
        super(myModelCheckpoint, self).__init__()
        self.monitor = monitor
        self.verbose = verbose
        self.filepath = filepath
        self.save_best_only = save_best_only
        self.save_weights_only = save_weights_only
        self.period = period
        self.epochs_since_last_save = 0
        self.remove_last_best = remove_last_best
        self.best_filepath = None

        if mode not in ['auto', 'min', 'max']:
            warnings.warn('ModelCheckpoint mode %s is unknown, '
                          'fallback to auto mode.' % (mode),
                          RuntimeWarning)
            mode = 'auto'

        if mode == 'min':
            self.monitor_op = np.less
            self.best = np.Inf
        elif mode == 'max':
            self.monitor_op = np.greater
            self.best = -np.Inf
        else:
            if 'acc' in self.monitor or self.monitor.startswith('fmeasure'):
                self.monitor_op = np.greater
                self.best = -np.Inf
            else:
                self.monitor_op = np.less
                self.best = np.Inf

    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.epochs_since_last_save += 1
        if self.epochs_since_last_save >= self.period:
            self.epochs_since_last_save = 0
            filepath = self.filepath.format(epoch=epoch, **logs)
            if self.save_best_only:
                current = logs.get(self.monitor)
                if current is None:
                    warnings.warn('Can save best model only with %s available, '
                                  'skipping.' % self.monitor, RuntimeWarning)
                else:
                    if self.monitor_op(current, self.best):
                        if self.verbose > 0:
                            print('Epoch %05d: %s improved from %0.5f to %0.5f,'
                                  ' saving model to %s'
                                  % (epoch, self.monitor, self.best,
                                     current, filepath))
                        self.best = current

                        if self.remove_last_best:
                            if self.best_filepath is not None:
                                if self.verbose > 0:
                                    print("Removing ", self.best_filepath)
                                utils.rm_file(self.best_filepath)
                            self.best_filepath = filepath

                        if self.save_weights_only:
                            self.model.save_weights(filepath, overwrite=True)
                        else:
                            self.model.save(filepath, overwrite=True)
                    else:
                        if self.verbose > 0:
                            print('Epoch %05d: %s did not improve' %
                                  (epoch, self.monitor))
            else:
                if self.verbose > 0:
                    print('Epoch %05d: saving model to %s' % (epoch, filepath))
                if self.save_weights_only:
                    self.model.save_weights(filepath, overwrite=True)
                else:
                    self.model.save(filepath, overwrite=True)


def get_checkpoint(sess_info, is_classification, save_best_only=True, tensor_board=False, monitor_also_loss=True):
    hdf5_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, 'hdf5')
    utils.mkdirs(hdf5_dir)

    if K.backend() != 'tensorflow':
        tensor_board = False

    res_cbs = []

    tensor_board_cp = None
    if tensor_board:
        tensor_board_dir = os.path.join(consts.OUTPUT_DIR, sess_info.out_dir, 'tensor_board')
        utils.mkdirs(tensor_board_dir)
        tensor_board_cp = TensorBoard(log_dir=tensor_board_dir, histogram_freq=1, write_graph=True)
        res_cbs.append(tensor_board_cp)

    if is_classification:
        monitor = "val_loss"
        extra_params = "_vacc{val_acc:.3f}"
    else:
        monitor = "val_loss"
        extra_params = ""

    hdf5_fname = sess_info.title + "_best_val_loss" + "_weights.e{epoch:03d}-loss{loss:.5f}-vloss{val_loss:.4f}" +\
        extra_params + ".hdf5"
    hdf5_full_path = os.path.join(hdf5_dir, hdf5_fname)
    model_cp = myModelCheckpoint(filepath=hdf5_full_path, verbose=0, save_best_only=save_best_only, monitor=monitor)
    res_cbs.append(model_cp)

    if monitor_also_loss:
        loss_hdf5_fname = sess_info.title + "_best_loss" + "_weights.e{epoch:03d}-loss{loss:.5f}-vloss{val_loss:.4f}" +\
                          extra_params + ".hdf5"
        loss_hdf5_full_path = os.path.join(hdf5_dir, loss_hdf5_fname)
        model_loss_cp = myModelCheckpoint(filepath=loss_hdf5_full_path, verbose=0, save_best_only=save_best_only,
                                          monitor="loss")
        res_cbs.append(model_loss_cp)

    return res_cbs


def angle_difference(x, y):
    """
    Calculate minimum difference between two angles.
    """
    return 180 - abs(abs(x - y) - 180)


# angle_error_regression() Taken from
# https://github.com/d4nst/RotNet/blob/master/utils.py
# See also its main website-
# https://d4nst.github.io/2017/01/12/image-orientation/
# https://github.com/d4nst/RotNet/blob/master/utils.py
def angle_error_regression(y_true, y_pred):
    """
    Calculate the mean difference between the true angles
    and the predicted angles. Each angle is represented
    as a float number between 0 and 1.
    """
    return K.mean(angle_difference(y_true * 360, y_pred * 360))


def meshNet_loss_OLD(y_true, y_pred):
    # TODO: Should find best alpha value
    alpha = 1.0

    # y_true_dbg = y_true
    # y_pred_dbg = y_pred
    y_true_dbg = K.print_tensor(y_true, 'y_true')
    y_pred_dbg = K.print_tensor(y_pred, 'y_pred')

    ret_xy_loss = K.square(y_pred_dbg[:, 0:2] - y_true_dbg[:, 0:2])
    ret_angle_loss = angle_difference(y_pred_dbg[:, 2:] * 360, y_true_dbg[:, 2:] * 360) / 360.0

    # alpha * K.square(angle_difference(y_pred[:, 2:] * 360, y_true[:, 2:] * 360) / 360.0), axis=-1)

    ret_xy_loss_dbg = K.print_tensor(ret_xy_loss, 'ret_xy_loss')
    ret_angle_loss_dbg = K.print_tensor(ret_angle_loss, 'ret_angle_loss')

    ret_loss = K.mean(ret_xy_loss_dbg + alpha * ret_angle_loss_dbg, axis=-1)
    ret_loss_dbg = K.print_tensor(ret_loss, 'ret_loss')

    return ret_loss_dbg


# https://stackoverflow.com/questions/37527832/keras-cost-function-for-cyclic-outputs
# def mean_squared_error(y_true, y_pred):
#     return K.mean(K.square(y_pred - y_true), axis=-1)
def cyclic_mean_squared_error(y_true, y_pred):
    return K.mean(K.minimum(K.square(y_pred - y_true),
                            K.minimum(K.square(y_pred - y_true + 1), K.square(y_pred - y_true - 1))),
                  axis=-1)


def meshNet_loss(y_true, y_pred):
    # TODO: Should find best alpha value
    # PoseNet Suggest alpha= test_xy_mean_error/test_angle_mean_error ~= 5/65
    alpha = 1.0

    ret_xy_loss = K.square(y_pred[:, 0:2] - y_true[:, 0:2])
    ret_angle_loss = K.minimum(K.square(y_pred[:, 2:] - y_true[:, 2:]),
                               K.minimum(K.square(y_pred[:, 2:] - y_true[:, 2:] + 1),
                               K.square(y_pred[:, 2:] - y_true[:, 2:] - 1)))

    # ret_xy_loss_dbg = K.print_tensor(ret_xy_loss, 'ret_xy_loss')
    # ret_angle_loss_dbg = K.print_tensor(ret_angle_loss, 'ret_angle_loss')
    # ret_loss = K.mean(K.concatenate((ret_xy_loss_dbg, alpha * ret_angle_loss_dbg)))

    ret_loss = K.mean(K.concatenate((ret_xy_loss, alpha * ret_angle_loss)))
    return ret_loss


def almost_VGG11_bn(image_shape, nb_outs, optimizer=None, loss=None):
    model_name = almost_VGG11_bn.__name__

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=image_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # model.add(Dense(4096))
    # model.add(PReLU(alpha_initializer=constant_init()))
    # model.add(Dense(4096))
    # model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Dense(512))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Dense(512))
    model.add(PReLU(alpha_initializer=constant_init()))

    model.add(Dense(nb_outs))

    if optimizer is None:
        optimizer = Adam()
    if loss is None:
        loss = "mean_squared_error"

    model.compile(loss=loss, optimizer=optimizer)

    return model, model_name


def almost_VGG11(image_shape, nb_outs, optimizer=None, loss=None):
    model_name = almost_VGG11.__name__

    model = Sequential()

    model.add(Conv2D(64, (3, 3), padding='same', input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(128, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(Conv2D(512, (3, 3)))
    # model.add(Activation('relu'))
    # model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Flatten())

    # model.add(Dense(4096))
    # model.add(PReLU(alpha_initializer=constant_init()))
    # model.add(Dense(4096))
    # model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Dense(512))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Dense(512))
    model.add(PReLU(alpha_initializer=constant_init()))

    model.add(Dense(nb_outs))

    if optimizer is None:
        optimizer = Adam()
    if loss is None:
        loss = "mean_squared_error"

    model.compile(loss=loss, optimizer=optimizer)

    return model, model_name


def reg_2_conv_relu_mp_2_conv_relu_dense_dense_bigger(image_shape, nb_outs, optimizer=None, loss=None):
    model_name = reg_2_conv_relu_mp_2_conv_relu_dense_dense_bigger.__name__

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=image_shape))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(32, (5, 5)))
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(32, (3, 3)))
    model.add(Activation('relu'))
    model.add(Conv2D(256, (3, 3)))
    model.add(Activation('relu'))

    model.add(Flatten())
    model.add(Dense(512))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Dense(nb_outs))

    if optimizer is None:
        optimizer = Adam()
    if loss is None:
        loss = "mean_squared_error"

    model.compile(loss=loss, optimizer=optimizer)

    return model, model_name


def reg_2_conv_prelu_mp_2_conv_prelu_dense_dense(image_shape, nb_outs, optimizer=None, loss=None):
    model_name = reg_2_conv_prelu_mp_2_conv_prelu_dense_dense.__name__

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=image_shape))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(MaxPooling2D(pool_size=(2, 2)))
    model.add(Conv2D(20, (5, 5)))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(MaxPooling2D(pool_size=(2, 2)))

    model.add(Conv2D(20, (3, 3)))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Conv2D(120, (3, 3)))
    model.add(PReLU(alpha_initializer=constant_init()))

    model.add(Flatten())
    model.add(Dense(120))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Dense(nb_outs))

    if optimizer is None:
        optimizer = RMSprop()
    if loss is None:
        loss = "mean_squared_error"

    model.compile(loss=loss, optimizer=optimizer)

    return model, model_name


def reg_2_conv_relu_mp_2_conv_relu_dense_dense(image_shape, nb_outs, optimizer=None, loss=None):
    model_name = reg_2_conv_relu_mp_2_conv_relu_dense_dense.__name__

    model = Sequential()

    model.add(Conv2D(32, (5, 5), padding='same', input_shape=image_shape))
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
    model.add(Dense(120))
    model.add(PReLU(alpha_initializer=constant_init()))
    model.add(Dense(nb_outs))

    if optimizer is None:
        optimizer = RMSprop()
    if loss is None:
        loss = "mean_squared_error"

    model.compile(loss=loss, optimizer=optimizer)

    return model, model_name


# FIXME: Add paper reference
def reg_2_conv_relu_mp_2_conv_relu_dense_dense_orig(image_shape, nb_outs, optimizer=None, loss=None):
    model_name = reg_2_conv_relu_mp_2_conv_relu_dense_dense_orig.__name__

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


# Based on SegNet Basic Model
# https://github.com/0bserver07/Keras-SegNet-Basic
def segNet(input_shape, output_shape, optimizer=None, loss=None):
    model_name = segNet.__name__

    def create_encoding_layers():
        kernel = 3
        filter_size = 64
        pad = 1
        pool_size = 2
        return [
            ZeroPadding2D(padding=(pad, pad)),
            Conv2D(filter_size, (kernel, kernel), padding='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),

            ZeroPadding2D(padding=(pad, pad)),
            Conv2D(128, (kernel, kernel), padding='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),

            ZeroPadding2D(padding=(pad, pad)),
            Conv2D(256, (kernel, kernel), padding='valid'),
            BatchNormalization(),
            Activation('relu'),
            MaxPooling2D(pool_size=(pool_size, pool_size)),

            ZeroPadding2D(padding=(pad, pad)),
            Conv2D(512, (kernel, kernel), padding='valid'),
            BatchNormalization(),
            Activation('relu'),
        ]

    def create_decoding_layers():
        kernel = 3
        filter_size = 64
        pad = 1
        pool_size = 2
        return [
            ZeroPadding2D(padding=(pad, pad)),
            Conv2D(512, (kernel, kernel), padding='valid'),
            BatchNormalization(),

            UpSampling2D(size=(pool_size, pool_size)),
            ZeroPadding2D(padding=(pad, pad)),
            Conv2D(256, (kernel, kernel), padding='valid'),
            BatchNormalization(),

            UpSampling2D(size=(pool_size, pool_size)),
            ZeroPadding2D(padding=(pad, pad)),
            Conv2D(128, (kernel, kernel), padding='valid'),
            BatchNormalization(),

            UpSampling2D(size=(pool_size, pool_size)),
            ZeroPadding2D(padding=(pad, pad)),
            Conv2D(filter_size, (kernel, kernel), padding='valid'),
            BatchNormalization(),
        ]

    model = Sequential()

    model.add(Layer(input_shape=input_shape))

    model.encoding_layers = create_encoding_layers()
    for l in model.encoding_layers:
        model.add(l)

    model.decoding_layers = create_decoding_layers()
    for l in model.decoding_layers:
        model.add(l)

    # What conv with (1, 1) does?
    # model.add(Conv2D(1, (1, 1), padding='valid'))

    model.add(Flatten())
    model.add(Dense(output_shape[0] * output_shape[1] * output_shape[2]))
    model.add(PReLU(alpha_initializer=constant_init()))

    model.add(Reshape(output_shape))

    # print(model.layers[-1].input_shape)
    # print(model.layers[-1].output_shape)

    if optimizer is None:
        optimizer = RMSprop()
    if loss is None:
        loss = "mean_squared_error"

    model.compile(loss=loss, optimizer=optimizer)

    return model, model_name


def create_googlenet(weights_path=None):
    # creates GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)

    input = Input(shape=(3, 224, 224))

    conv1_7x7_s2 = Conv2D(64, 7, 7, subsample=(2, 2), border_mode='same', activation='relu', name='conv1/7x7_s2',
                          W_regularizer=l2(0.0002))(input)

    conv1_zero_pad = ZeroPadding2D(padding=(1, 1))(conv1_7x7_s2)

    pool1_helper = PoolHelper()(conv1_zero_pad)

    pool1_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool1/3x3_s2')(
        pool1_helper)

    pool1_norm1 = LRN(name='pool1/norm1')(pool1_3x3_s2)

    conv2_3x3_reduce = Conv2D(64, 1, 1, border_mode='same', activation='relu', name='conv2/3x3_reduce',
                              W_regularizer=l2(0.0002))(pool1_norm1)

    conv2_3x3 = Conv2D(192, 3, 3, border_mode='same', activation='relu', name='conv2/3x3',
                       W_regularizer=l2(0.0002))(conv2_3x3_reduce)

    conv2_norm2 = LRN(name='conv2/norm2')(conv2_3x3)

    conv2_zero_pad = ZeroPadding2D(padding=(1, 1))(conv2_norm2)

    pool2_helper = PoolHelper()(conv2_zero_pad)

    pool2_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool2/3x3_s2')(
        pool2_helper)

    inception_3a_1x1 = Conv2D(64, 1, 1, border_mode='same', activation='relu', name='inception_3a/1x1',
                              W_regularizer=l2(0.0002))(pool2_3x3_s2)

    inception_3a_3x3_reduce = Conv2D(96, 1, 1, border_mode='same', activation='relu',
                                     name='inception_3a/3x3_reduce', W_regularizer=l2(0.0002))(pool2_3x3_s2)

    inception_3a_3x3 = Conv2D(128, 3, 3, border_mode='same', activation='relu', name='inception_3a/3x3',
                              W_regularizer=l2(0.0002))(inception_3a_3x3_reduce)

    inception_3a_5x5_reduce = Conv2D(16, 1, 1, border_mode='same', activation='relu',
                                     name='inception_3a/5x5_reduce', W_regularizer=l2(0.0002))(pool2_3x3_s2)

    inception_3a_5x5 = Conv2D(32, 5, 5, border_mode='same', activation='relu', name='inception_3a/5x5',
                              W_regularizer=l2(0.0002))(inception_3a_5x5_reduce)

    inception_3a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_3a/pool')(
        pool2_3x3_s2)

    inception_3a_pool_proj = Conv2D(32, 1, 1, border_mode='same', activation='relu',
                                    name='inception_3a/pool_proj', W_regularizer=l2(0.0002))(inception_3a_pool)

    inception_3a_output = merge([inception_3a_1x1, inception_3a_3x3, inception_3a_5x5, inception_3a_pool_proj],
                                mode='concat', concat_axis=1, name='inception_3a/output')

    inception_3b_1x1 = Conv2D(128, 1, 1, border_mode='same', activation='relu', name='inception_3b/1x1',
                              W_regularizer=l2(0.0002))(inception_3a_output)

    inception_3b_3x3_reduce = Conv2D(128, 1, 1, border_mode='same', activation='relu',
                                     name='inception_3b/3x3_reduce', W_regularizer=l2(0.0002))(
        inception_3a_output)

    inception_3b_3x3 = Conv2D(192, 3, 3, border_mode='same', activation='relu', name='inception_3b/3x3',
                              W_regularizer=l2(0.0002))(inception_3b_3x3_reduce)

    inception_3b_5x5_reduce = Conv2D(32, 1, 1, border_mode='same', activation='relu',
                                     name='inception_3b/5x5_reduce', W_regularizer=l2(0.0002))(
        inception_3a_output)

    inception_3b_5x5 = Conv2D(96, 5, 5, border_mode='same', activation='relu', name='inception_3b/5x5',
                              W_regularizer=l2(0.0002))(inception_3b_5x5_reduce)

    inception_3b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_3b/pool')(
        inception_3a_output)

    inception_3b_pool_proj = Conv2D(64, 1, 1, border_mode='same', activation='relu',
                                    name='inception_3b/pool_proj', W_regularizer=l2(0.0002))(inception_3b_pool)

    inception_3b_output = merge([inception_3b_1x1, inception_3b_3x3, inception_3b_5x5, inception_3b_pool_proj],
                                mode='concat', concat_axis=1, name='inception_3b/output')

    inception_3b_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_3b_output)

    pool3_helper = PoolHelper()(inception_3b_output_zero_pad)

    pool3_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool3/3x3_s2')(
        pool3_helper)

    inception_4a_1x1 = Conv2D(192, 1, 1, border_mode='same', activation='relu', name='inception_4a/1x1',
                              W_regularizer=l2(0.0002))(pool3_3x3_s2)

    inception_4a_3x3_reduce = Conv2D(96, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4a/3x3_reduce', W_regularizer=l2(0.0002))(pool3_3x3_s2)

    inception_4a_3x3 = Conv2D(208, 3, 3, border_mode='same', activation='relu', name='inception_4a/3x3',
                              W_regularizer=l2(0.0002))(inception_4a_3x3_reduce)

    inception_4a_5x5_reduce = Conv2D(16, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4a/5x5_reduce', W_regularizer=l2(0.0002))(pool3_3x3_s2)

    inception_4a_5x5 = Conv2D(48, 5, 5, border_mode='same', activation='relu', name='inception_4a/5x5',
                              W_regularizer=l2(0.0002))(inception_4a_5x5_reduce)

    inception_4a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_4a/pool')(
        pool3_3x3_s2)

    inception_4a_pool_proj = Conv2D(64, 1, 1, border_mode='same', activation='relu',
                                    name='inception_4a/pool_proj', W_regularizer=l2(0.0002))(inception_4a_pool)

    inception_4a_output = merge([inception_4a_1x1, inception_4a_3x3, inception_4a_5x5, inception_4a_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4a/output')

    loss1_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss1/ave_pool')(inception_4a_output)

    loss1_conv = Conv2D(128, 1, 1, border_mode='same', activation='relu', name='loss1/conv',
                        W_regularizer=l2(0.0002))(loss1_ave_pool)

    loss1_flat = Flatten()(loss1_conv)

    loss1_fc = Dense(1024, activation='relu', name='loss1/fc', W_regularizer=l2(0.0002))(loss1_flat)

    loss1_drop_fc = Dropout(0.7)(loss1_fc)

    loss1_classifier = Dense(1000, name='loss1/classifier', W_regularizer=l2(0.0002))(loss1_drop_fc)

    loss1_classifier_act = Activation('softmax')(loss1_classifier)

    inception_4b_1x1 = Conv2D(160, 1, 1, border_mode='same', activation='relu', name='inception_4b/1x1',
                              W_regularizer=l2(0.0002))(inception_4a_output)

    inception_4b_3x3_reduce = Conv2D(112, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4b/3x3_reduce', W_regularizer=l2(0.0002))(
        inception_4a_output)

    inception_4b_3x3 = Conv2D(224, 3, 3, border_mode='same', activation='relu', name='inception_4b/3x3',
                              W_regularizer=l2(0.0002))(inception_4b_3x3_reduce)

    inception_4b_5x5_reduce = Conv2D(24, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4b/5x5_reduce', W_regularizer=l2(0.0002))(
        inception_4a_output)

    inception_4b_5x5 = Conv2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4b/5x5',
                              W_regularizer=l2(0.0002))(inception_4b_5x5_reduce)

    inception_4b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_4b/pool')(
        inception_4a_output)

    inception_4b_pool_proj = Conv2D(64, 1, 1, border_mode='same', activation='relu',
                                    name='inception_4b/pool_proj', W_regularizer=l2(0.0002))(inception_4b_pool)

    inception_4b_output = merge([inception_4b_1x1, inception_4b_3x3, inception_4b_5x5, inception_4b_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4b_output')

    inception_4c_1x1 = Conv2D(128, 1, 1, border_mode='same', activation='relu', name='inception_4c/1x1',
                              W_regularizer=l2(0.0002))(inception_4b_output)

    inception_4c_3x3_reduce = Conv2D(128, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4c/3x3_reduce', W_regularizer=l2(0.0002))(
        inception_4b_output)

    inception_4c_3x3 = Conv2D(256, 3, 3, border_mode='same', activation='relu', name='inception_4c/3x3',
                              W_regularizer=l2(0.0002))(inception_4c_3x3_reduce)

    inception_4c_5x5_reduce = Conv2D(24, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4c/5x5_reduce', W_regularizer=l2(0.0002))(
        inception_4b_output)

    inception_4c_5x5 = Conv2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4c/5x5',
                              W_regularizer=l2(0.0002))(inception_4c_5x5_reduce)

    inception_4c_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_4c/pool')(
        inception_4b_output)

    inception_4c_pool_proj = Conv2D(64, 1, 1, border_mode='same', activation='relu',
                                    name='inception_4c/pool_proj', W_regularizer=l2(0.0002))(inception_4c_pool)

    inception_4c_output = merge([inception_4c_1x1, inception_4c_3x3, inception_4c_5x5, inception_4c_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4c/output')

    inception_4d_1x1 = Conv2D(112, 1, 1, border_mode='same', activation='relu', name='inception_4d/1x1',
                              W_regularizer=l2(0.0002))(inception_4c_output)

    inception_4d_3x3_reduce = Conv2D(144, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4d/3x3_reduce', W_regularizer=l2(0.0002))(
        inception_4c_output)

    inception_4d_3x3 = Conv2D(288, 3, 3, border_mode='same', activation='relu', name='inception_4d/3x3',
                              W_regularizer=l2(0.0002))(inception_4d_3x3_reduce)

    inception_4d_5x5_reduce = Conv2D(32, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4d/5x5_reduce', W_regularizer=l2(0.0002))(
        inception_4c_output)

    inception_4d_5x5 = Conv2D(64, 5, 5, border_mode='same', activation='relu', name='inception_4d/5x5',
                              W_regularizer=l2(0.0002))(inception_4d_5x5_reduce)

    inception_4d_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_4d/pool')(
        inception_4c_output)

    inception_4d_pool_proj = Conv2D(64, 1, 1, border_mode='same', activation='relu',
                                    name='inception_4d/pool_proj', W_regularizer=l2(0.0002))(inception_4d_pool)

    inception_4d_output = merge([inception_4d_1x1, inception_4d_3x3, inception_4d_5x5, inception_4d_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4d/output')

    loss2_ave_pool = AveragePooling2D(pool_size=(5, 5), strides=(3, 3), name='loss2/ave_pool')(inception_4d_output)

    loss2_conv = Conv2D(128, 1, 1, border_mode='same', activation='relu', name='loss2/conv',
                        W_regularizer=l2(0.0002))(loss2_ave_pool)

    loss2_flat = Flatten()(loss2_conv)

    loss2_fc = Dense(1024, activation='relu', name='loss2/fc', W_regularizer=l2(0.0002))(loss2_flat)

    loss2_drop_fc = Dropout(0.7)(loss2_fc)

    loss2_classifier = Dense(1000, name='loss2/classifier', W_regularizer=l2(0.0002))(loss2_drop_fc)

    loss2_classifier_act = Activation('softmax')(loss2_classifier)

    inception_4e_1x1 = Conv2D(256, 1, 1, border_mode='same', activation='relu', name='inception_4e/1x1',
                              W_regularizer=l2(0.0002))(inception_4d_output)

    inception_4e_3x3_reduce = Conv2D(160, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4e/3x3_reduce', W_regularizer=l2(0.0002))(
        inception_4d_output)

    inception_4e_3x3 = Conv2D(320, 3, 3, border_mode='same', activation='relu', name='inception_4e/3x3',
                              W_regularizer=l2(0.0002))(inception_4e_3x3_reduce)

    inception_4e_5x5_reduce = Conv2D(32, 1, 1, border_mode='same', activation='relu',
                                     name='inception_4e/5x5_reduce', W_regularizer=l2(0.0002))(
        inception_4d_output)

    inception_4e_5x5 = Conv2D(128, 5, 5, border_mode='same', activation='relu', name='inception_4e/5x5',
                              W_regularizer=l2(0.0002))(inception_4e_5x5_reduce)

    inception_4e_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_4e/pool')(
        inception_4d_output)

    inception_4e_pool_proj = Conv2D(128, 1, 1, border_mode='same', activation='relu',
                                    name='inception_4e/pool_proj', W_regularizer=l2(0.0002))(inception_4e_pool)

    inception_4e_output = merge([inception_4e_1x1, inception_4e_3x3, inception_4e_5x5, inception_4e_pool_proj],
                                mode='concat', concat_axis=1, name='inception_4e/output')

    inception_4e_output_zero_pad = ZeroPadding2D(padding=(1, 1))(inception_4e_output)

    pool4_helper = PoolHelper()(inception_4e_output_zero_pad)

    pool4_3x3_s2 = MaxPooling2D(pool_size=(3, 3), strides=(2, 2), border_mode='valid', name='pool4/3x3_s2')(
        pool4_helper)

    inception_5a_1x1 = Conv2D(256, 1, 1, border_mode='same', activation='relu', name='inception_5a/1x1',
                              W_regularizer=l2(0.0002))(pool4_3x3_s2)

    inception_5a_3x3_reduce = Conv2D(160, 1, 1, border_mode='same', activation='relu',
                                     name='inception_5a/3x3_reduce', W_regularizer=l2(0.0002))(pool4_3x3_s2)

    inception_5a_3x3 = Conv2D(320, 3, 3, border_mode='same', activation='relu', name='inception_5a/3x3',
                              W_regularizer=l2(0.0002))(inception_5a_3x3_reduce)

    inception_5a_5x5_reduce = Conv2D(32, 1, 1, border_mode='same', activation='relu',
                                     name='inception_5a/5x5_reduce', W_regularizer=l2(0.0002))(pool4_3x3_s2)

    inception_5a_5x5 = Conv2D(128, 5, 5, border_mode='same', activation='relu', name='inception_5a/5x5',
                              W_regularizer=l2(0.0002))(inception_5a_5x5_reduce)

    inception_5a_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_5a/pool')(
        pool4_3x3_s2)

    inception_5a_pool_proj = Conv2D(128, 1, 1, border_mode='same', activation='relu',
                                    name='inception_5a/pool_proj', W_regularizer=l2(0.0002))(inception_5a_pool)

    inception_5a_output = merge([inception_5a_1x1, inception_5a_3x3, inception_5a_5x5, inception_5a_pool_proj],
                                mode='concat', concat_axis=1, name='inception_5a/output')

    inception_5b_1x1 = Conv2D(384, 1, 1, border_mode='same', activation='relu', name='inception_5b/1x1',
                              W_regularizer=l2(0.0002))(inception_5a_output)

    inception_5b_3x3_reduce = Conv2D(192, 1, 1, border_mode='same', activation='relu',
                                     name='inception_5b/3x3_reduce', W_regularizer=l2(0.0002))(
        inception_5a_output)

    inception_5b_3x3 = Conv2D(384, 3, 3, border_mode='same', activation='relu', name='inception_5b/3x3',
                              W_regularizer=l2(0.0002))(inception_5b_3x3_reduce)

    inception_5b_5x5_reduce = Conv2D(48, 1, 1, border_mode='same', activation='relu',
                                     name='inception_5b/5x5_reduce', W_regularizer=l2(0.0002))(
        inception_5a_output)

    inception_5b_5x5 = Conv2D(128, 5, 5, border_mode='same', activation='relu', name='inception_5b/5x5',
                              W_regularizer=l2(0.0002))(inception_5b_5x5_reduce)

    inception_5b_pool = MaxPooling2D(pool_size=(3, 3), strides=(1, 1), border_mode='same', name='inception_5b/pool')(
        inception_5a_output)

    inception_5b_pool_proj = Conv2D(128, 1, 1, border_mode='same', activation='relu',
                                    name='inception_5b/pool_proj', W_regularizer=l2(0.0002))(inception_5b_pool)

    inception_5b_output = merge([inception_5b_1x1, inception_5b_3x3, inception_5b_5x5, inception_5b_pool_proj],
                                mode='concat', concat_axis=1, name='inception_5b/output')

    pool5_7x7_s1 = AveragePooling2D(pool_size=(7, 7), strides=(1, 1), name='pool5/7x7_s2')(inception_5b_output)

    loss3_flat = Flatten()(pool5_7x7_s1)

    pool5_drop_7x7_s1 = Dropout(0.4)(loss3_flat)

    loss3_classifier = Dense(1000, name='loss3/classifier', W_regularizer=l2(0.0002))(pool5_drop_7x7_s1)

    loss3_classifier_act = Activation('softmax', name='prob')(loss3_classifier)

    googlenet = Model(input=input, output=[loss1_classifier_act, loss2_classifier_act, loss3_classifier_act])

    if weights_path:
        googlenet.load_weights(weights_path)

    return googlenet


def main():
    print("Entered")

    model, model_name = segNet((600, 800, 1), (1024, 1024, 1))
    print (model_name)

    print("Done")

if __name__ == '__main__':
    main()
