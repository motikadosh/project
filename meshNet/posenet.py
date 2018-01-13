# File Taken from
# https://github.com/kentsommer/keras-posenet

# from scipy.misc import imread, imresize

from keras.layers import Input, Dense, Convolution2D
from keras.layers import MaxPooling2D, AveragePooling2D
from keras.layers import ZeroPadding2D, Dropout, Flatten
from keras.layers import merge, Reshape, Activation, BatchNormalization
# from keras.utils.np_utils import convert_kernel
from keras import backend as K
from keras.models import Model
from keras.optimizers import Adam, Adadelta
import tensorflow as tf
import numpy as np

# import h5py
# import math


def euc_loss1x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.3 * lx)

def euc_loss1q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (150 * lq)

def euc_loss2x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (0.3 * lx)

def euc_loss2q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (150 * lq)

def euc_loss3x(y_true, y_pred):
    lx = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (1 * lx)

def euc_loss3q(y_true, y_pred):
    lq = K.sqrt(K.sum(K.square(y_true[:,:] - y_pred[:,:]), axis=1, keepdims=True))
    return (500 * lq)


# FIXME: Seems I am returning 2 columns from this loss function. What does it do???
# FIXME: I.e. NOT scalar for each sample as in any other example I could find.
# FIXME: E.g. see above losses or Keras example-
# FIXME: def mean_squared_error(y_true, y_pred):
# FIXME:     return K.mean(K.square(y_pred - y_true), axis=-1)
def cyclic_loss_1(y_true, y_pred):
    lcyc = K.minimum(K.square(y_pred - y_true),
                     K.minimum(K.square(y_pred - y_true + 1),
                     K.square(y_pred - y_true - 1)))
    return (0.3) * lcyc

def cyclic_loss_2(y_true, y_pred):
    lcyc = K.minimum(K.square(y_pred - y_true),
                     K.minimum(K.square(y_pred - y_true + 1),
                     K.square(y_pred - y_true - 1)))
    return (0.3) * lcyc

def cyclic_loss_3(y_true, y_pred):
    lcyc = K.minimum(K.square(y_pred - y_true),
                     K.minimum(K.square(y_pred - y_true + 1),
                     K.square(y_pred - y_true - 1)))
    return (1.0) * lcyc

# TODO: READ http://www.boris-belousov.net/2016/12/01/quat-dist/

# From: 3D Pose Regression using Convolutional Neural Networks - https://arxiv.org/abs/1708.05628
# LOSS(R, R') = abs(arccos( 0.5*(trace(transpose(R) * R')-1) ))
def rot_mat_loss1(y_true, y_pred):
    K.abs( tf.acos( 0.5*() ))
    pass


def my_print_tensor(x, message=''):
    import tensorflow as tf
    return tf.Print(x, [x], message, summarize=1024)

# Debug
#def cyclic_loss_3(y_true, y_pred):
#
#    y_true_dbg = my_print_tensor(y_true, 'y_true')
#    y_pred_dbg = my_print_tensor(y_pred, 'y_pred')
#    # y_pred_dbg = K.print_tensor(y_pred, 'y_pred')
#
#    a = K.square(y_pred_dbg - y_true_dbg)
#    b = K.square(y_pred_dbg - y_true_dbg + 1)
#    c = K.square(y_pred_dbg - y_true_dbg - 1)
#
#    a_dbg = my_print_tensor(a, 'a')
#    b_dbg = my_print_tensor(b, 'b')
#    c_dbg = my_print_tensor(c, 'c')
#
#    min_bc = K.minimum(b_dbg, c_dbg)
#    min_bc_dbg = my_print_tensor(min_bc, 'min_bc')
#
#    min_abc = K.minimum(a_dbg, min_bc_dbg)
#    min_abc_dbg = my_print_tensor(min_abc, 'min_abc')
#
#    lcyc = min_abc_dbg
#
#    lcyc_dbg = my_print_tensor(lcyc, 'lcyc')
#
#    return (1.0) * lcyc_dbg


def create_posenet(image_shape=(224, 224, 3), xy_nb_outs=2, rot_nb_outs=2, weights_path=None,
                   tune=False):
    # creates Posenet from GoogLeNet a.k.a. Inception v1 (Szegedy, 2015)
    with tf.device('/cpu:0'):
        input = Input(shape=image_shape)
        
        conv1 = Convolution2D(64,7,7,subsample=(2,2),border_mode='same',activation='relu',name='conv1')(input)
        
        pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same',name='pool1')(conv1)

        norm1 = BatchNormalization(axis=3, name='norm1')(pool1)
        
        reduction2 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='reduction2')(norm1)
        
        conv2 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='conv2')(reduction2)

        norm2 = BatchNormalization(axis=3, name='norm2')(conv2)
        
        pool2 = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='valid',name='pool2')(norm2)

        icp1_reduction1 = Convolution2D(96,1,1,border_mode='same',activation='relu',name='icp1_reduction1')(pool2)

        icp1_out1 = Convolution2D(128,3,3,border_mode='same',activation='relu',name='icp1_out1')(icp1_reduction1)
        

        icp1_reduction2 = Convolution2D(16,1,1,border_mode='same',activation='relu',name='icp1_reduction2')(pool2)

        icp1_out2 = Convolution2D(32,5,5,border_mode='same',activation='relu',name='icp1_out2')(icp1_reduction2)
        

        icp1_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp1_pool')(pool2)

        icp1_out3 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp1_out3')(icp1_pool)

       
        icp1_out0 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp1_out0')(pool2)

        
        icp2_in = merge([icp1_out0, icp1_out1, icp1_out2, icp1_out3],mode='concat',concat_axis=3,name='icp2_in')


        
        


        icp2_reduction1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp2_reduction1')(icp2_in)

        icp2_out1 = Convolution2D(192,3,3,border_mode='same',activation='relu',name='icp2_out1')(icp2_reduction1)
        
       
        icp2_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp2_reduction2')(icp2_in)

        icp2_out2 = Convolution2D(96,5,5,border_mode='same',activation='relu',name='icp2_out2')(icp2_reduction2)


        icp2_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp2_pool')(icp2_in)

        icp2_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp2_out3')(icp2_pool)


        icp2_out0 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp2_out0')(icp2_in)

        
        icp2_out = merge([icp2_out0, icp2_out1, icp2_out2, icp2_out3],mode='concat',concat_axis=3,name='icp2_out')






        icp3_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same',name='icp3_in')(icp2_out)

        icp3_reduction1 = Convolution2D(96,1,1,border_mode='same',activation='relu',name='icp3_reduction1')(icp3_in)

        icp3_out1 = Convolution2D(208,3,3,border_mode='same',activation='relu',name='icp3_out1')(icp3_reduction1)


        icp3_reduction2 = Convolution2D(16,1,1,border_mode='same',activation='relu',name='icp3_reduction2')(icp3_in)

        icp3_out2 = Convolution2D(48,5,5,border_mode='same',activation='relu',name='icp3_out2')(icp3_reduction2)
        

        icp3_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp3_pool')(icp3_in)

        icp3_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp3_out3')(icp3_pool)

        
        icp3_out0 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='icp3_out0')(icp3_in)
        
        
        icp3_out = merge([icp3_out0, icp3_out1, icp3_out2, icp3_out3],mode='concat',concat_axis=3,name='icp3_out')





        # Moti- Change pool_size -> (2,2)
        # cls1_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid',name='cls1_pool')(icp3_out)
        cls1_pool = AveragePooling2D(pool_size=(2, 2), border_mode='valid', name='cls1_pool')(icp3_out)
        cls1_reduction_pose = Convolution2D(128,1,1,border_mode='same',activation='relu',name='cls1_reduction_pose')(cls1_pool)


        cls1_fc1_flat = Flatten()(cls1_reduction_pose)
        
        cls1_fc1_pose = Dense(1024,activation='relu',name='cls1_fc1_pose')(cls1_fc1_flat)

        cls1_fc_pose_xyz = Dense(xy_nb_outs,name='cls1_fc_pose_xyz')(cls1_fc1_pose)
        
        cls1_fc_pose_wpqr = Dense(rot_nb_outs, name='cls1_fc_pose_wpqr')(cls1_fc1_pose)





        
        icp4_reduction1 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='icp4_reduction1')(icp3_out)

        icp4_out1 = Convolution2D(224,3,3,border_mode='same',activation='relu',name='icp4_out1')(icp4_reduction1)

        
        icp4_reduction2 = Convolution2D(24,1,1,border_mode='same',activation='relu',name='icp4_reduction2')(icp3_out)

        icp4_out2 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='icp4_out2')(icp4_reduction2)


        icp4_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp4_pool')(icp3_out)

        icp4_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp4_out3')(icp4_pool)


        icp4_out0 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='icp4_out0')(icp3_out)

        
        icp4_out = merge([icp4_out0, icp4_out1, icp4_out2, icp4_out3],mode='concat',concat_axis=3,name='icp4_out')






        icp5_reduction1 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp5_reduction1')(icp4_out)

        icp5_out1 = Convolution2D(256,3,3,border_mode='same',activation='relu',name='icp5_out1')(icp5_reduction1)


        icp5_reduction2 = Convolution2D(24,1,1,border_mode='same',activation='relu',name='icp5_reduction2')(icp4_out)

        icp5_out2 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='icp5_out2')(icp5_reduction2)


        icp5_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp5_pool')(icp4_out)

        icp5_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp5_out3')(icp5_pool)


        icp5_out0 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp5_out0')(icp4_out)

        
        icp5_out = merge([icp5_out0, icp5_out1, icp5_out2, icp5_out3],mode='concat',concat_axis=3,name='icp5_out')






        icp6_reduction1 = Convolution2D(144,1,1,border_mode='same',activation='relu',name='icp6_reduction1')(icp5_out)

        icp6_out1 = Convolution2D(288,3,3,border_mode='same',activation='relu',name='icp6_out1')(icp6_reduction1)

        
        icp6_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp6_reduction2')(icp5_out)

        icp6_out2 = Convolution2D(64,5,5,border_mode='same',activation='relu',name='icp6_out2')(icp6_reduction2)

        
        icp6_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp6_pool')(icp5_out)

        icp6_out3 = Convolution2D(64,1,1,border_mode='same',activation='relu',name='icp6_out3')(icp6_pool)


        icp6_out0 = Convolution2D(112,1,1,border_mode='same',activation='relu',name='icp6_out0')(icp5_out)

        
        icp6_out = merge([icp6_out0, icp6_out1, icp6_out2, icp6_out3],mode='concat',concat_axis=3,name='icp6_out')
        




        # Moti- Change pool_size -> (2,2)
        # cls2_pool = AveragePooling2D(pool_size=(5,5),strides=(3,3),border_mode='valid',name='cls2_pool')(icp6_out)
        cls2_pool = AveragePooling2D(pool_size=(2,2),border_mode='valid',name='cls2_pool')(icp6_out)
        cls2_reduction_pose = Convolution2D(128,1,1,border_mode='same',activation='relu',name='cls2_reduction_pose')(cls2_pool)


        cls2_fc1_flat = Flatten()(cls2_reduction_pose)

        cls2_fc1 = Dense(1024,activation='relu',name='cls2_fc1')(cls2_fc1_flat)
        
        cls2_fc_pose_xyz = Dense(xy_nb_outs,name='cls2_fc_pose_xyz')(cls2_fc1)
        
        cls2_fc_pose_wpqr = Dense(rot_nb_outs, name='cls2_fc_pose_wpqr')(cls2_fc1)






        icp7_reduction1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='icp7_reduction1')(icp6_out)

        icp7_out1 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='icp7_out1')(icp7_reduction1)


        icp7_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp7_reduction2')(icp6_out)

        icp7_out2 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='icp7_out2')(icp7_reduction2)


        icp7_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp7_pool')(icp6_out)

        icp7_out3 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp7_out3')(icp7_pool)

        
        icp7_out0 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='icp7_out0')(icp6_out)
        

        icp7_out = merge([icp7_out0, icp7_out1, icp7_out2, icp7_out3],mode='concat',concat_axis=3,name='icp7_out')

        
        


        
        icp8_in = MaxPooling2D(pool_size=(3,3),strides=(2,2),border_mode='same',name='icp8_in')(icp7_out)
        
        icp8_reduction1 = Convolution2D(160,1,1,border_mode='same',activation='relu',name='icp8_reduction1')(icp8_in)

        icp8_out1 = Convolution2D(320,3,3,border_mode='same',activation='relu',name='icp8_out1')(icp8_reduction1)


        icp8_reduction2 = Convolution2D(32,1,1,border_mode='same',activation='relu',name='icp8_reduction2')(icp8_in)

        icp8_out2 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='icp8_out2')(icp8_reduction2)


        icp8_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp8_pool')(icp8_in)

        icp8_out3 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp8_out3')(icp8_pool)

        
        icp8_out0 = Convolution2D(256,1,1,border_mode='same',activation='relu',name='icp8_out0')(icp8_in)
        
        icp8_out = merge([icp8_out0, icp8_out1, icp8_out2, icp8_out3],mode='concat',concat_axis=3,name='icp8_out')
        





        icp9_reduction1 = Convolution2D(192,1,1,border_mode='same',activation='relu',name='icp9_reduction1')(icp8_out)

        icp9_out1 = Convolution2D(384,3,3,border_mode='same',activation='relu',name='icp9_out1')(icp9_reduction1)


        icp9_reduction2 = Convolution2D(48,1,1,border_mode='same',activation='relu',name='icp9_reduction2')(icp8_out)

        icp9_out2 = Convolution2D(128,5,5,border_mode='same',activation='relu',name='icp9_out2')(icp9_reduction2)


        icp9_pool = MaxPooling2D(pool_size=(3,3),strides=(1,1),border_mode='same',name='icp9_pool')(icp8_out)

        icp9_out3 = Convolution2D(128,1,1,border_mode='same',activation='relu',name='icp9_out3')(icp9_pool)

        
        icp9_out0 = Convolution2D(384,1,1,border_mode='same',activation='relu',name='icp9_out0')(icp8_out)
        
        icp9_out = merge([icp9_out0, icp9_out1, icp9_out2, icp9_out3],mode='concat',concat_axis=3,name='icp9_out')
        




        # Moti- Skip pooling
        # cls3_pool = AveragePooling2D(pool_size=(7,7),strides=(1,1),border_mode='valid',name='cls3_pool')(icp9_out)
        cls3_pool = icp9_out
        cls3_fc1_flat = Flatten()(cls3_pool)

        cls3_fc1_pose = Dense(2048,activation='relu',name='cls3_fc1_pose')(cls3_fc1_flat)

        
        cls3_fc_pose_xyz = Dense(xy_nb_outs,name='cls3_fc_pose_xyz')(cls3_fc1_pose)
        
        cls3_fc_pose_wpqr = Dense(rot_nb_outs, name='cls3_fc_pose_wpqr')(cls3_fc1_pose)
        





        posenet = Model(input=input, output=[cls1_fc_pose_xyz, cls1_fc_pose_wpqr,
                                             cls2_fc_pose_xyz, cls2_fc_pose_wpqr,
                                             cls3_fc_pose_xyz, cls3_fc_pose_wpqr])
    
    if tune:
        if weights_path:
            weights_data = np.load(weights_path).item()
            for layer in posenet.layers:
                if layer.name in weights_data.keys():
                    layer_weights = weights_data[layer.name]
                    layer.set_weights((layer_weights['weights'], layer_weights['biases']))
            print("FINISHED SETTING THE WEIGHTS!")

    return posenet


def posenet_train(image_shape, xy_nb_outs, rot_nb_outs, optimizer=None, loss=None):
    model_name = posenet_train.__name__

    # Train model - GoogLeNet (Trained on Places)
    model = create_posenet(image_shape=image_shape, xy_nb_outs=xy_nb_outs, rot_nb_outs=rot_nb_outs)

    if rot_nb_outs == 2:   # 'angle'
        rot_loss_1 = cyclic_loss_1
        rot_loss_2 = cyclic_loss_2
        rot_loss_3 = cyclic_loss_3

    elif rot_nb_outs == 4:  # 'quaternion'
        rot_loss_1 = euc_loss1q
        rot_loss_2 = euc_loss2q
        rot_loss_3 = euc_loss3q

    elif rot_nb_outs == 9:  # 'matrix'
        rot_loss_1 = rot_mat_loss1
        rot_loss_2 = rot_mat_loss1
        rot_loss_3 = rot_mat_loss1

    else:
        raise Exception("Unknown rot_nb_outs value:", rot_nb_outs)

    if optimizer is None:
        optimizer = Adam(lr=0.001, clipvalue=1.5)  # Original
        # optimizer = Adadelta()
    if loss is None:
        loss = {'cls1_fc_pose_xyz': euc_loss1x, 'cls1_fc_pose_wpqr': rot_loss_1,
                'cls2_fc_pose_xyz': euc_loss2x, 'cls2_fc_pose_wpqr': rot_loss_2,
                'cls3_fc_pose_xyz': euc_loss3x, 'cls3_fc_pose_wpqr': rot_loss_3}

    model.compile(optimizer=optimizer, loss=loss)

    return model, model_name
