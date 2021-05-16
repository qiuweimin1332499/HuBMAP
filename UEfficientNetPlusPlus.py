#pip install tf_slim
#pip install tf-nightly

import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dense, Dropout, \
    BatchNormalization, LeakyReLU, Add, Softmax
from tensorflow.keras.applications import EfficientNetB4

def standard_unit(blockInput, stage, nb_filter=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, nb_filter, (3, 3))
    x = convolution_block(x, nb_filter, (3, 3), activation=False)
    x = Add()([x, blockInput])
    return x

def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x



def UNet_pp(input_shape=(None, None, 3), deep_supervision=True):  # Unet
    '''
     1-1---> 1-2 ---> 1-3 ---> 1-4 --->1-5
        \   /   \   /    \    /   \   /
         2-1 --->2-2 ---> 2-3 --->2-4
           \    /   \    /   \   /
            3-1 ---> 3-2 ---> 3-3
              \     /   \    /
                4-1---> 4-2
                  \     /
                    5-1
    '''
    backbone = EfficientNetB4(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape)

    input_tensor = backbone.input
    nb_filter = [16, 32, 64, 128, 256]

    conv4 = backbone.layers[342].output
    conv3 = backbone.layers[153].output
    conv2 = backbone.layers[92].output
    conv1 = backbone.layers[33].output
    conv0 = backbone.layers[4].output

    conv0 = Conv2D(nb_filter[0], (3, 3), activation=None, padding="same")(conv0)
    conv1_1 = standard_unit(conv0, stage='stage_11', nb_filter=nb_filter[0])
    pool1 = MaxPooling2D([2, 2])(conv1_1)

    conv2_1 = tf.concat([pool1, conv1], axis=-1)
    conv2_1 = Conv2D(nb_filter[1], (3, 3), activation=None, padding="same")(conv2_1)
    conv2_1 = standard_unit(conv2_1, stage='stage_21', nb_filter=nb_filter[1])
    pool2 = MaxPooling2D((2, 2))(conv2_1)

    conv3_1 = tf.concat([pool2, conv2], axis=-1)
    conv3_1 = Conv2D(nb_filter[2], (3, 3), activation=None, padding="same")(conv3_1)
    conv3_1 = standard_unit(conv3_1, stage='stage_31', nb_filter=nb_filter[2])
    pool3 = MaxPooling2D((2, 2))(conv3_1)

    conv4_1 = tf.concat([pool3, conv3], axis=-1)
    conv4_1 = Conv2D(nb_filter[3], (3, 3), activation=None, padding="same")(conv4_1)
    conv4_1 = standard_unit(conv4_1, stage='stage_41', nb_filter=nb_filter[3])
    pool4 = MaxPooling2D((2, 2))(conv4_1)

    conv5_1 = tf.concat([pool4, conv4], axis=-1)
    conv5_1 = Conv2D(nb_filter[4], (3, 3), activation=None, padding="same")(conv5_1)
    conv5_1 = standard_unit(conv5_1, stage='stage_51', nb_filter=nb_filter[4])

    up1_2 = Conv2DTranspose(filters=nb_filter[0], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv2_1)
    conv1_2 = tf.concat([conv1_1, up1_2], axis=-1)
    conv1_2 = Conv2D(nb_filter[0], (3, 3), activation=None, padding="same")(conv1_2)
    conv1_2 = standard_unit(conv1_2, stage='stage_12', nb_filter=nb_filter[0])

    up2_2 =  Conv2DTranspose(filters=nb_filter[1], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv3_1)
    conv2_2 = tf.concat([conv2_1, up2_2], axis=-1)
    conv2_2 = Conv2D(nb_filter[1], (3, 3), activation=None, padding="same")(conv2_2)
    conv2_2 = standard_unit(conv2_2, stage='stage_22', nb_filter=nb_filter[1])

    up3_2 =  Conv2DTranspose(filters=nb_filter[2], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv4_1)
    conv3_2 = tf.concat([conv3_1, up3_2], axis=-1)
    conv3_2 = Conv2D(nb_filter[2], (3, 3), activation=None, padding="same")(conv3_2)
    conv3_2 = standard_unit(conv3_2, stage='stage_32', nb_filter=nb_filter[2])

    up4_2 =  Conv2DTranspose(filters=nb_filter[3], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv5_1)
    conv4_2 = tf.concat([conv4_1, up4_2], axis=-1)
    conv4_2 = Conv2D(nb_filter[3], (3, 3), activation=None, padding="same")(conv4_2)
    conv4_2 = standard_unit(conv4_2, stage='stage_42', nb_filter=nb_filter[3])

    up1_3 =  Conv2DTranspose(filters=nb_filter[0], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv2_2)
    conv1_3 = tf.concat([conv1_1, conv1_2, up1_3], axis=-1)
    conv1_3 = Conv2D(nb_filter[0], (3, 3), activation=None, padding="same")(conv1_3)
    conv1_3 = standard_unit(conv1_3, stage='stage_13', nb_filter=nb_filter[0])

    up2_3 =  Conv2DTranspose(filters=nb_filter[1], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv3_2)
    conv2_3 = tf.concat([conv2_1, conv2_2, up2_3], axis=-1)
    conv2_3 = Conv2D(nb_filter[1], (3, 3), activation=None, padding="same")(conv2_3)
    conv2_3 = standard_unit(conv2_3, stage='stage_23', nb_filter=nb_filter[1])

    up3_3 =  Conv2DTranspose(filters=nb_filter[2], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv4_2)
    conv3_3 = tf.concat([conv3_1, conv3_2, up3_3], axis=-1)
    conv3_3 = Conv2D(nb_filter[2], (3, 3), activation=None, padding="same")(conv3_3)
    conv3_3 = standard_unit(conv3_3, stage='stage_33', nb_filter=nb_filter[2])

    up1_4 =  Conv2DTranspose(filters=nb_filter[0], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv2_3)
    conv1_4 = tf.concat([conv1_1, conv1_2, conv1_3, up1_4], axis=-1)
    conv1_4 = Conv2D(nb_filter[0], (3, 3), activation=None, padding="same")(conv1_4)
    conv1_4 = standard_unit(conv1_4, stage='stage_14', nb_filter=nb_filter[0])

    up2_4 =  Conv2DTranspose(filters=nb_filter[1], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv3_3)
    conv2_4 = tf.concat([conv2_1, conv2_2, conv2_3, up2_4], axis=-1)
    conv2_4 = Conv2D(nb_filter[1], (3, 3), activation=None, padding="same")(conv2_4)
    conv2_4 = standard_unit(conv2_4, stage='stage_24', nb_filter=nb_filter[1])

    up1_5 =  Conv2DTranspose(filters=nb_filter[2], kernel_size=(3, 3), strides=(2, 2), padding="same")(conv2_4)
    conv1_5 = tf.concat([conv1_1, conv1_2, conv1_3, conv1_4, up1_5], axis=-1)
    conv1_5 = Conv2D(nb_filter[0], (3, 3), activation=None, padding="same")(conv1_5)
    conv1_5 = standard_unit(conv1_5, stage='stage_15', nb_filter=nb_filter[0])

    nestnet_output_1 = Conv2D(1, [1, 1], padding="same", activation="sigmoid")(conv1_2)
    nestnet_output_1 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding="same")(nestnet_output_1)
    nestnet_output_2 = Conv2D(1, [1, 1], padding="same", activation="sigmoid")(conv1_3)
    nestnet_output_2 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding="same")(nestnet_output_2)
    nestnet_output_3 = Conv2D(1, [1, 1], padding="same", activation="sigmoid")(conv1_4)
    nestnet_output_3 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding="same")(nestnet_output_3)
    nestnet_output_4 = Conv2D(1, [1, 1], padding="same", activation="sigmoid")(conv1_5)
    nestnet_output_4 = Conv2DTranspose(filters=1, kernel_size=(3, 3), strides=(2, 2), padding="same")(nestnet_output_4)

    if deep_supervision:
        h_deconv_concat = tf.concat([nestnet_output_1, nestnet_output_2, nestnet_output_3, nestnet_output_4], axis=-1)
        h_deconv_concat = Conv2D(filters=1, kernel_size=3, padding="same", activation="tanh")(h_deconv_concat)
        model = Model(input_tensor, h_deconv_concat)
        return model
    else:
        model = Model(input_tensor, nestnet_output_4)
        return model


def crop_and_concat(x1, x2):
    with tf.name_scope("crop_and_concat"):
        x1_shape = tf.shape(x1)
        x2_shape = tf.shape(x2)
        # offsets for the top left corner of the crop
        offsets = [0, (x1_shape[1] - x2_shape[1]) // 2, (x1_shape[2] - x2_shape[2]) // 2, 0]
        size = [-1, x2_shape[1], x2_shape[2], -1]
        x1_crop = tf.slice(x1, offsets, size)
        return tf.concat([x1_crop, x2], axis=-1)


def upsample(x, num_outputs, batch_size=4):
    pool_size = 2
    input_wh = int(x.shape[1])
    in_channels = int(x.shape[-1])
    output_shape = (batch_size, input_wh * 2, input_wh * 2, num_outputs)
    deconv_filter = tf.Variable(tf.random.truncated_normal([pool_size, pool_size, num_outputs, in_channels], stddev=0.02))
    deconv = tf.nn.conv2d_transpose(x, deconv_filter, output_shape, strides=[1, pool_size, pool_size, 1])
    return deconv