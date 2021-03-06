import os
import cv2
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Conv2D, MaxPooling2D, UpSampling2D, Conv2DTranspose, Dense, Dropout, \
    BatchNormalization, LeakyReLU, Add
from tensorflow.keras.applications import EfficientNetB4
from deformable_conv_layer import DeformableConvLayer
#pip install tf-nightly

def UEfficientNet(input_shape=(None, None, 3), dropout_rate=0.1):
    backbone = EfficientNetB4(weights='imagenet',
                              include_top=False,
                              input_shape=input_shape)

    input = backbone.input
    start_neurons = 8
    #print('conv4:')
    #print(tf.shape(conv4))
    #for i in range(len(backbone.layers)):   # get tensor output shape
        #print(i)
        #print (tf.shape(backbone.layers[i].output))
    conv4 = backbone.layers[342].output
    conv4 = LeakyReLU(alpha=0.1)(conv4)
    pool4 = MaxPooling2D((2, 2))(conv4)
    pool4 = Dropout(dropout_rate)(pool4)

    # Middle
    convm = Conv2D(start_neurons * 32, (3, 3), activation=None, padding="same", name='conv_middle')(pool4)
    convm = residual_block(convm, start_neurons * 32)
    convm = residual_block(convm, start_neurons * 32)
    convm = LeakyReLU(alpha=0.1)(convm)

    deconv4 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(convm)
    deconv4_up1 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4)
    deconv4_up2 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up1)
    deconv4_up3 = Conv2DTranspose(start_neurons * 16, (3, 3), strides=(2, 2), padding="same")(deconv4_up2)
    uconv4 = tf.concat([deconv4, conv4], axis=-1)
    uconv4 = Dropout(dropout_rate)(uconv4)
    uconv4 = Conv2D(start_neurons * 16, (3, 3), activation=None, padding="same")(uconv4)
    uconv4 = residual_block(uconv4, start_neurons * 16)
    uconv4 = LeakyReLU(alpha=0.1)(uconv4)  # conv1_2

    deconv3 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(uconv4)
    deconv3_up1 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3)
    deconv3_up2 = Conv2DTranspose(start_neurons * 8, (3, 3), strides=(2, 2), padding="same")(deconv3_up1)
    conv3 = backbone.layers[153].output
    uconv3 = tf.concat([deconv3, deconv4_up1, conv3], axis=-1)
    uconv3 = Dropout(dropout_rate)(uconv3)
    uconv3 = Conv2D(start_neurons * 8, (3, 3), activation=None, padding="same")(uconv3)
    uconv3 = residual_block(uconv3, start_neurons * 8)
    uconv3 = LeakyReLU(alpha=0.1)(uconv3)

    deconv2 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(uconv3)
    deconv2_up1 = Conv2DTranspose(start_neurons * 4, (3, 3), strides=(2, 2), padding="same")(deconv2)
    conv2 = backbone.layers[92].output
    uconv2 = tf.concat([deconv2, deconv3_up1, deconv4_up2, conv2], axis=-1)
    uconv2 = Dropout(0.1)(uconv2)
    uconv2 = Conv2D(start_neurons * 4, (3, 3), activation=None, padding="same")(uconv2)
    uconv2 = residual_block(uconv2, start_neurons * 4)
    uconv2 = LeakyReLU(alpha=0.1)(uconv2)

    deconv1 = Conv2DTranspose(start_neurons * 2, (3, 3), strides=(2, 2), padding="same")(uconv2)
    conv1 = backbone.layers[33].output
    uconv1 = tf.concat([deconv1, deconv2_up1, deconv3_up2, deconv4_up3, conv1], axis=-1)
    uconv1 = Dropout(0.1)(uconv1)
    uconv1 = Conv2D(start_neurons * 2, (3, 3), activation=None, padding="same")(uconv1)
    uconv1 = residual_block(uconv1, start_neurons * 2)
    uconv1 = LeakyReLU(alpha=0.1)(uconv1)

    uconv0 = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv1)
    uconv0 = Dropout(0.1)(uconv0)
    uconv0 = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(uconv0)
    uconv0 = residual_block(uconv0, start_neurons * 1)
    uconv0 = LeakyReLU(alpha=0.1)(uconv0)
    uconv0 = Dropout(dropout_rate / 2)(uconv0)

    output_layer = Conv2DTranspose(start_neurons * 1, (3, 3), strides=(2, 2), padding="same")(uconv0)
    output_layer = Dropout(0.1)(output_layer)
    output_layer = Conv2D(start_neurons * 1, (3, 3), activation=None, padding="same")(output_layer)
    output_layer = residual_block(output_layer, start_neurons * 1)
    output_layer = LeakyReLU(alpha=0.1)(output_layer)
    output_layer = Dropout(dropout_rate / 2)(output_layer)
    output_layer = Conv2D(1, (1, 1), padding="same", activation="sigmoid")(output_layer)
    #print('output_layer:')
    #print(tf.shape(output_layer))
    model = Model(input, output_layer)
    #model.name = 'unetb4'

    return model


def convolution_block(x, filters, size, strides=(1, 1), padding='same', activation=True):
    x = Conv2D(filters, size, strides=strides, padding=padding)(x)
    x = BatchNormalization()(x)
    if activation == True:
        x = LeakyReLU(alpha=0.1)(x)
    return x


def residual_block(blockInput, num_filters=16):
    x = LeakyReLU(alpha=0.1)(blockInput)
    x = BatchNormalization()(x)
    blockInput = BatchNormalization()(blockInput)
    x = convolution_block(x, num_filters, (3, 3))
    x = convolution_block(x, num_filters, (3, 3), activation=False)
    x = Add()([x, blockInput])
    return x


if __name__ == '__main__':
    img_size = 512
    threshold_best = 0.5
    model = UEfficientNet(input_shape=(img_size, img_size, 3), dropout_rate=0.3)
    model.load_weights('./weights_binary/keras.model')
    test_path = "D:/xxx/unet_plus/road_test/val/imgs/"
    out = "D:/xxx/unet_plus/predict_binary/"
    imgs = os.listdir(test_path)
    count = len(imgs)
    testGene = testGenerator(test_path)
    results = model.predict_generator(testGene, count, verbose=1)
    for i in range(count):
        out_path = os.path.join(out, imgs[i])
        im = results[i]
        im[im > threshold_best] = 255
        im[im <= threshold_best] = 0
        cv2.imwrite(out_path, im)
