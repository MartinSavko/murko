#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# based on F. Chollet's https://keras.io/examples/vision/oxford_pets_image_segmentation/
# Model based on The One Hundred Layers Tiramisu: Fully convolutional DenseNets for Semantic Segmentation, arXiv:1611.09326
# With main difference being use of SeparableConv2D instead of Conv2D and
# using GroupNormalization instead of BatchNormalization. Plus using
# additional Weight standardization (based on Qiao et al. Micro-Batch
# Training with Batch-Channel Normalization and Weight Standardization
# arXiv:1903.10520v2)

from dataset_loader import (
    MultiTargetDataset,
    get_dynamic_batch_size,
    size_differs,
    get_transposed_img_and_target,
    get_transformed_img_and_target,
    get_hierarchical_mask_from_target,
)

from utils import efficient_resize

import tensorflow as tf

from tensorflow import keras
from keras import regularizers, initializers

from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model
from keras.preprocessing.image import save_img, load_img, img_to_array, array_to_img
import os
import sys
import time
import math
import zmq
import glob
import numpy as np
import random
import re
import pickle
import traceback
import pylab
import simplejpeg
import copy

import scipy.ndimage as ndi

# directory = "images_and_labels_augmented"
# img_size = (1024, 1360)
# model_img_size = (512, 512)
# num_classes = 1
# batch_size = 8

"""
dataset composition:
      notion         fraction_label      fraction_total

     crystal:           0.051084,           0.023338
 loop_inside:           0.118100,           0.053953
        loop:           0.294535,           0.134557
        stem:           0.062713,           0.028650
         pin:           0.035046,           0.016011
   capillary:           0.004127,           0.001885
         ice:           0.026989,           0.012330
  foreground:           0.407406,           0.186121
  background:           1.781519,           0.813879


"""
# loss_weights_from_stats =\
# {'crystal': 8,
# 'loop_inside': 3.5,
# 'loop': 1.5,
# 'stem': 6.5,
# 'pin': 5.0,
# 'capillary': 1.,
# 'ice': 1.,
# 'foreground': 1.0,
# 'click': 1.}

"""
total pixels 1805946880 (1.806G)
total foreground 319001744 (0.319G, 0.1766 of all)

         notion  fraction_label  fraction_total         weight_label         weight_total
        crystal       0.1173            0.0207                8.5                48.3
    loop_inside       0.2485            0.0439                4.0                22.8
           loop       0.6427            0.1135                1.6                 8.8
           stem       0.1876            0.0331                5.3                30.2
            pin       0.1242            0.0219                8.0                45.6
      capillary       0.0102            0.0018               98.4               557.0
            ice       0.0630            0.0111               15.9                89.9
     foreground       1.0000            0.1766                1.0                 5.7
          total       5.6612            1.0000                0.2                 1.0
"""

params = {
    "binary_segmentation": {"loss": "binary_focal_crossentropy", "metrics": "BIoU"},
    "distance_transform": {
        "loss": "mean_squared_error",
        "metrics": "mean_absolute_error",
    },
    "bounding_box": {"loss": "mean_squared_error", "metrics": "mean_absolute_error"},
    "inner_points": {"loss": "mean_squared_error", "metrics": "mean_absolute_error"},
    "extreme_points": {"loss": "mean_squared_error", "metrics": "mean_absolute_error"},
    "eigen_points": {"loss": "mean_squared_error", "metrics": "mean_absolute_error"},
    "encoded_shape": {"loss": "mean_squared_error", "metrics": "mean_absolute_error"},
    "categorical_segmentation": {
        "loss": "categorical_focal_crossentropy",
        "metrics": "MeanIoU",
    },
    "encoder": {"loss": "mean_squared_error", "metrics": "mean_absolute_error"},
    "point": {"loss": "mean_squared_error", "metrics": "mean_absolute_error"},
}

networks = {
    "fcdn103": {
        "growth_rate": 16,
        "layers_scheme": [4, 5, 7, 10, 12],
        "bottleneck": 15,
    },
    "fcdn67": {"growth_rate": 16, "layers_scheme": [5] * 5, "bottleneck": 5},
    "fcdn56": {"growth_rate": 12, "layers_scheme": [4] * 5, "bottleneck": 4},
}

loss_weights_from_stats = {
    "crystal": 8.5,
    "loop_inside": 4.0,
    "loop": 1.6,
    "stem": 5.3,
    "pin": 8.0,
    "capillary": 1.0,
    "ice": 15.9,
    "foreground": 1.0,
    "click": 1.0,
}


class UpsampleLike(keras.layers.Layer):
    """
    Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    based on https://github.com/xuannianz/keras-fcos.git
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return resize_images(
            source, (target_shape[1], target_shape[2]), method="nearest"
        )

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)


class WSConv2D_keras(keras.layers.Conv2D):
    """https://github.com/joe-siyuan-qiao/WeightStandardization"""

    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(*args, **kwargs)
        self.eps = 1.0e-5
        self.std = False

    def standardize_kernel(self, kernel):
        original_dtype = kernel.dtype

        mean = keras.ops.mean(kernel, axis=(0, 1, 2), keepdims=True)
        kernel = kernel - mean

        if self.std:
            std = keras.ops.std(kernel, axis=[0, 1, 2], keepdims=True)
            std = std + self.eps
            kernel = kernel / std

        kernel = keras.ops.cast(kernel, dtype=original_dtype)

        return kernel

    def call(self, inputs):
        self.kernel.assign(self.standardize_kernel(self.kernel))
        return super().call(inputs)


class WSSeparableConv2D_keras(keras.layers.SeparableConv2D):
    """https://github.com/joe-siyuan-qiao/WeightStandardization"""

    def __init__(self, *args, **kwargs):
        super(WSSeparableConv2D, self).__init__(*args, **kwargs)
        self.eps = 1.0e-5
        self.std = False

    def standardize_kernel(self, kernel):
        original_dtype = kernel.dtype

        mean = keras.ops.mean(kernel, axis=(0, 1, 2), keepdims=True)
        kernel = kernel - mean

        if self.std:
            std = keras.ops.std(kernel, axis=[0, 1, 2], keepdims=True)
            std = std + self.eps
            kernel = kernel / std

        kernel = keras.ops.cast(kernel, dtype=original_dtype)

        return kernel

    def call(self, inputs):
        self.pointwise_kernel.assign(self.standardize_kernel(self.pointwise_kernel))
        self.depthwise_kernel.assign(self.standardize_kernel(self.depthwise_kernel))

        return super().call(inputs)


class WSConv2D(tf.keras.layers.Conv2D):
    """https://github.com/joe-siyuan-qiao/WeightStandardization"""

    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(*args, **kwargs)
        self.eps = 1.0e-5
        self.std = False

    def standardize_kernel(self, kernel):
        original_dtype = kernel.dtype

        mean = tf.math.reduce_mean(kernel, axis=(0, 1, 2), keepdims=True)
        kernel = kernel - mean

        if self.std:
            std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
            std = std + tf.constant(self.eps, dtype=std.dtype)
            kernel = kernel / std

        kernel = tf.cast(kernel, dtype=original_dtype)

        return kernel

    def call(self, inputs):
        self.kernel.assign(self.standardize_kernel(self.kernel))
        return super().call(inputs)


class WSSeparableConv2D(tf.keras.layers.SeparableConv2D):
    """https://github.com/joe-siyuan-qiao/WeightStandardization"""

    def __init__(self, *args, **kwargs):
        super(WSSeparableConv2D, self).__init__(*args, **kwargs)
        self.eps = 1.0e-5
        self.std = False

    def standardize_kernel(self, kernel):
        original_dtype = kernel.dtype

        mean = tf.math.reduce_mean(kernel, axis=(0, 1, 2), keepdims=True)
        kernel = kernel - mean

        if self.std:
            std = tf.math.reduce_std(kernel, axis=[0, 1, 2], keepdims=True)
            std = std + tf.constant(self.eps, dtype=std.dtype)
            kernel = kernel / std

        kernel = tf.cast(kernel, dtype=original_dtype)

        return kernel

    def call(self, inputs):
        self.pointwise_kernel.assign(self.standardize_kernel(self.pointwise_kernel))
        self.depthwise_kernel.assign(self.standardize_kernel(self.depthwise_kernel))

        return super().call(inputs)


def find_number_of_groups(c, g):
    w, r = divmod(c, g)
    if r == 0:
        return g
    else:
        return find_number_of_groups(c, g - 1)


def get_kernel_initializer(kernel_initializer):
    return getattr(initializers, kernel_initializer)()


def get_kernel_regularizer(kernel_regularizer, weight_decay):
    if weight_decay == 0.0:
        return None
    return getattr(regularizers, kernel_regularizer)(weight_decay)


def get_convolutional_layer(
    x,
    convolution_type,
    filters,
    filter_size=3,
    padding="same",
    activation="relu",
    kernel_initializer="he_normal",
    kernel_regularizer="l2",
    weight_decay=1e-4,
    use_bias=False,
    weight_standardization=True,
):
    kwargs = {
        "kernel_initializer": get_kernel_initializer(kernel_initializer),
        "kernel_regularizer": get_kernel_regularizer(kernel_regularizer, weight_decay),
    }
    if weight_standardization:
        if convolution_type == "SeparableConv2D":
            x = WSSeparableConv2D(
                filters, filter_size, padding=padding, use_bias=use_bias, **kwargs
            )(x)
        elif convolution_type == "Conv2D":
            x = WSConv2D(
                filters, filter_size, padding=padding, use_bias=use_bias, **kwargs
            )(x)
    else:
        x = getattr(keras.layers, convolution_type)(
            filters, filter_size, padding=padding, use_bias=use_bias, **kwargs
        )(x)
    return x


def get_tiramisu_layer(
    x,
    filters,
    filter_size=3,
    padding="same",
    activation="relu",
    convolution_type="Conv2D",
    kernel_initializer="he_normal",
    kernel_regularizer="l2",
    weight_decay=1e-4,
    use_bias=False,
    normalization_type="GroupNormalization",
    bn_momentum=0.9,
    bn_epsilon=1.1e-5,
    gn_groups=16,
    dropout_rate=0.2,
    weight_standardization=True,
    invert=True,
):
    if invert:
        x = get_convolutional_layer(
            x,
            convolution_type,
            filters,
            filter_size,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            weight_decay=weight_decay,
            weight_standardization=weight_standardization,
        )
        x = get_normalization_layer(
            x,
            normalization_type,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            gn_groups=gn_groups,
        )
        x = keras.layers.Activation(activation=activation)(x)
    else:
        x = get_normalization_layer(
            x,
            normalization_type,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            gn_groups=gn_groups,
        )
        x = keras.layers.Activation(activation=activation)(x)
        x = get_convolutional_layer(
            x,
            convolution_type,
            filters,
            filter_size,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=kernel_regularizer,
            weight_decay=weight_decay,
            weight_standardization=weight_standardization,
        )

    if dropout_rate:
        x = keras.layers.Dropout(dropout_rate)(x)
    return x


def get_dense_block(
    x,
    filters,
    number_of_layers,
    padding="same",
    activation="relu",
    convolution_type="Conv2D",
    dropout_rate=0.2,
    up=False,
    use_bias=False,
    kernel_initializer="he_normal",
    kernel_regularizer="l2",
    weight_decay=1e-4,
    bn_momentum=0.9,
    bn_epsilon=1.1e-5,
    normalization_type="GroupNormalization",
    weight_standardization=True,
):
    block_to_upsample = [x]
    for l in range(number_of_layers):
        la = get_tiramisu_layer(
            x,
            filters,
            padding=padding,
            activation=activation,
            convolution_type=convolution_type,
            dropout_rate=dropout_rate,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            weight_decay=weight_decay,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            normalization_type=normalization_type,
            weight_standardization=weight_standardization,
        )
        block_to_upsample.append(la)
        x = keras.layers.Concatenate(axis=3)([x, la])
    return x, block_to_upsample


def get_transition_down(
    x,
    filters,
    filter_size=(1, 1),
    padding="same",
    activation="relu",
    convolution_type="Conv2D",
    dropout_rate=0.2,
    use_bias=False,
    kernel_initializer="he_normal",
    kernel_regularizer="l2",
    weight_decay=1e-4,
    bn_momentum=0.9,
    bn_epsilon=1.1e-5,
    pool_size=2,
    strides=2,
    normalization_type="GroupNormalization",
    weight_standardization=True,
):
    if filter_size == (1, 1) or filter_size == 1:
        convolution_type = "Conv2D"
    x = get_tiramisu_layer(
        x,
        filters,
        filter_size=filter_size,
        padding=padding,
        activation=activation,
        convolution_type=convolution_type,
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        kernel_regularizer=kernel_regularizer,
        weight_decay=weight_decay,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        normalization_type=normalization_type,
        weight_standardization=weight_standardization,
    )
    x = keras.layers.MaxPooling2D(
        pool_size=pool_size, strides=strides, padding=padding
    )(x)
    return x


def get_transition_up(
    skip_connection,
    block_to_upsample,
    filters,
    padding="same",
    activation="relu",
    kernel_initializer="he_normal",
    kernel_regularizer="l2",
    weight_decay=1e-4,
    **kwargs,
):
    x = keras.layers.Concatenate(axis=3)(block_to_upsample[1:])
    x = keras.layers.Conv2DTranspose(
        filters,
        kernel_size=3,
        strides=2,
        padding=padding,
        activation=activation,
        kernel_regularizer=get_kernel_regularizer(kernel_regularizer, weight_decay),
    )(x)
    x = keras.layers.Concatenate(axis=3)([x, skip_connection])
    return x


def get_normalization_layer(
    x, normalization_type, bn_momentum=0.9, bn_epsilon=1.1e-5, gn_groups=16
):
    if normalization_type in ["BN", "BatchNormalization"]:
        x = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(x)
    elif normalization_type in ["GN", "GroupNormalization"]:
        x = keras.layers.GroupNormalization(
            groups=find_number_of_groups(x.shape[-1], gn_groups)
        )(x)
    elif normalization_type == "BCN":
        x = keras.layers.GroupNormalization(
            groups=find_number_of_groups(x.shape[-1], gn_groups)
        )(x)
        x = keras.layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(x)
    return x


def get_num_segmentation_classes(heads):
    num_segmentation_classes = 0
    for head in heads:
        if head["type"] == "binary_segmentation":
            num_segmentation_classes += 1
    return num_segmentation_classes


def get_uncompiled_tiramisu(
    nfilters=48,
    growth_rate=16,
    layers_scheme=[4, 5, 7, 10, 12],
    bottleneck=15,
    activation="relu",
    convolution_type="SeparableConv2D",
    padding="same",
    last_convolution=False,
    dropout_rate=0.2,
    weight_standardization=True,
    model_img_size=(None, None),
    input_channels=3,
    use_bias=False,
    kernel_initializer="he_normal",
    kernel_regularizer="l2",
    weight_decay=1e-4,
    heads=[
        {"name": "crystal", "type": "binary_segmentation"},
        {"name": "loop_inside", "type": "binary_segmentation"},
        {"name": "loop", "type": "binary_segmentation"},
        {"name": "stem", "type": "binary_segmentation"},
        {"name": "pin", "type": "binary_segmentation"},
        {"name": "foreground", "type": "binary_segmentation"},
    ],
    verbose=False,
    name="model",
    normalization_type="GroupNormalization",
    gn_groups=16,
    bn_momentum=0.9,
    bn_epsilon=1.1e-5,
    input_dropout=0.0,
):
    print("get_uncompiled_tiramisu heads", heads)
    boilerplate = {
        "activation": activation,
        "convolution_type": convolution_type,
        "padding": padding,
        "dropout_rate": dropout_rate,
        "use_bias": use_bias,
        "kernel_initializer": kernel_initializer,
        "kernel_regularizer": kernel_regularizer,
        "weight_decay": weight_decay,
        "normalization_type": normalization_type,
        "weight_standardization": weight_standardization,
    }

    inputs = keras.layers.Input(shape=(model_img_size) + (input_channels,))

    nfilters_start = nfilters

    if input_dropout > 0.0:
        x = keras.layers.Dropout(dropout_rate=input_dropout)(inputs)
    else:
        x = inputs

    x = get_tiramisu_layer(
        x,
        nfilters,
        filter_size=3,
        padding=padding,
        activation=activation,
        convolution_type="Conv2D",
        dropout_rate=dropout_rate,
        use_bias=use_bias,
        kernel_initializer=kernel_initializer,
        weight_decay=weight_decay,
        bn_momentum=bn_momentum,
        bn_epsilon=bn_epsilon,
        normalization_type=normalization_type,
        weight_standardization=weight_standardization,
    )

    _skips = []

    # DOWN
    for l, number_of_layers in enumerate(layers_scheme):
        x, block_to_upsample = get_dense_block(
            x, growth_rate, number_of_layers, **boilerplate
        )
        _skips.append(x)
        nfilters += number_of_layers * growth_rate
        x = get_transition_down(x, nfilters, **boilerplate)
        if verbose:
            print("layer:", l, number_of_layers, "shape:", x.shape)

    # BOTTLENECK
    x, block_to_upsample = get_dense_block(x, growth_rate, bottleneck, **boilerplate)
    if verbose:
        print("bottleneck:", l, number_of_layers, "shape:", x.shape)
    _skips = _skips[::-1]
    extended_layers_scheme = layers_scheme + [bottleneck]
    extended_layers_scheme.reverse()

    # UP
    for l, number_of_layers in enumerate(layers_scheme[::-1]):
        n_filters_keep = growth_rate * extended_layers_scheme[l]
        if verbose:
            print("n_filters_keep", n_filters_keep)
        x = get_transition_up(_skips[l], block_to_upsample, n_filters_keep)
        x_up, block_to_upsample = get_dense_block(
            x, growth_rate, number_of_layers, **boilerplate
        )
        if verbose:
            print(
                "layer:",
                l,
                number_of_layers,
                "shape:",
                x.shape,
                "x_up.shape",
                x_up.shape,
            )

    # OUTPUTS
    outputs = []
    regression_neck = None
    num_segmentation_classes = get_num_segmentation_classes(heads)
    for head in heads:
        if (
            head["type"] == "binary_segmentation"
            or head["type"] == "click_segmentation"
            or head["name"] == "encoder"
        ):
            output = keras.layers.Conv2D(
                1,
                1,
                activation="sigmoid",
                padding="same",
                dtype="float32",
                name=head["name"],
            )(x_up)
        elif (
            head["type"] == "categorical_segmentation" and num_segmentation_classes > 0
        ):
            output = keras.layers.Conv2D(
                num_segmentation_classes + 1,
                1,
                activation="softmax",
                padding="same",
                dtype="float32",
                name=head["name"],
            )(x_up)

            # output = get_convolutional_layer(x_up, 'Conv2D', 1, filter_size=1, padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, weight_decay=weight_decay, weight_standardization=weight_standardization, activation="sigmoid", dtype="float32", name=head['name'])

        elif head["type"] == "regression":
            if regression_neck is None:
                # regression_neck = keras.layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype="float32", name=head['name'])(x_up)
                regression_neck = get_tiramisu_layer(
                    x_up, 1, 1, activation="sigmoid", convolution_type="Conv2D"
                )
                batch_size, input_shape_y, input_shape_w, channels = K.shape(
                    regression_neck
                ).numpy
                target_shape = (batch_size, 224, 224, channels)
                target_placeholder = K.placeholder(shape=target_shape)
                regression_neck = UpsampleLike(name="resize_regression")(
                    [regression_neck, target_placeholder]
                )
                regression_neck = keras.layers.Flatten()(regression_neck)
                regression_neck = keras.layers.Dropout(dropout_rate)(regression_neck)
                # regression_neck = get_transition_down(regression_neck, 512, strides=11, pool_size=11)
                # regression_neck = keras.layers.GlobalMaxPool2D()(regression_neck)
                # regression_neck = keras.layers.Flatten()(regression_neck)
            output = keras.layers.Dense(
                3, activation="sigmoid", dtype="float32", name=head["name"]
            )(regression_neck)
        outputs.append(output)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def predict_multihead(
    to_predict=None,
    image_paths=None,
    base="/nfs/data2/Martin/Research/murko",
    model_name="fcdn103_256x320_loss_weights.h5",
    directory="images_and_labels",
    nimages=-1,
    batch_size=16,
    model_img_size=(224, 224),
    augment=False,
    threshold=0.5,
    train=False,
    split=0.2,
    target=False,
    model=None,
    save=True,
    prefix="prefix",
):
    _start = time.time()
    if model is None:
        model = keras.models.load_model(
            os.path.join(base, model_name),
            custom_objects={
                "WSConv2D": WSConv2D,
                "WSSeparableConv2D": WSSeparableConv2D,
            },
        )
        print("model loaded in %.4f seconds" % (time.time() - _start))

    notions = [
        layer.name
        for layer in model.layers[-10:]
        if isinstance(layer, keras.layers.Conv2D)
    ]
    notion_indices = dict([(notion, notions.index(notion)) for notion in notions])
    notion_indices["click"] = -1

    model_img_size = get_closest_working_img_size(model_img_size)
    print("model_img_size will be", model_img_size)

    all_image_paths = []
    if to_predict is None:
        train_paths, val_paths = get_training_and_validation_datasets(
            directory, split=split
        )
        if train:
            to_predict = train_paths
        else:
            to_predict = val_paths
    elif (
        not isinstance(to_predict, list)
        and not isinstance(to_predict, np.ndarray)
        and os.path.isdir(to_predict)
    ):
        to_predict = glob.glob(os.path.join(to_predict, "*.jpg"))
        all_image_paths = [os.path.realpath(t) for t in to_predict]
    elif (
        not isinstance(to_predict, list)
        and not isinstance(to_predict, np.ndarray)
        and os.path.isfile(to_predict)
    ):
        print("we seem to have received a single imagename to do our analysis on")
        all_image_paths.append(os.path.realpath(to_predict))
        to_predict = np.expand_dims(get_img(to_predict, size=model_img_size), 0)
    elif isinstance(to_predict, bytes) and simplejpeg.is_jpeg(to_predict):
        img_array = simplejpeg.decode_jpeg(to_predict)
        to_predict = np.expand_dims(img_array, 0)
    elif isinstance(to_predict, list):
        if simplejpeg.is_jpeg(to_predict[0]):
            to_predict = [simplejpeg.decode_jpeg(jpeg) for jpeg in to_predict]
        elif os.path.isfile(to_predict[0]):
            all_image_paths = [img_path for img_path in to_predict]
            to_predict = [
                img_to_array(load_img(img, target_size=model_img_size), dtype="float32")
                for img in to_predict
            ]
        if isinstance(to_predict[0], np.ndarray):
            to_predict = [img.astype("float32") / 255.0 for img in to_predict]
        if size_differs(to_predict[0].shape[:2], model_img_size):
            to_predict = [
                efficient_resize(img, model_img_size, anti_aliasing=True)
                for img in to_predict
            ]
        to_predict = np.array(to_predict)
    elif isinstance(to_predict, np.ndarray):
        if len(to_predict.shape) == 3:
            to_predict = np.expand_dims(to_predict, 0)
        if size_differs(to_predict[0].shape[:2], model_img_size):
            to_predict = np.array(
                [
                    efficient_resize(img, model_img_size, anti_aliasing=True)
                    for img in to_predict
                ]
            )
    print(
        "all images (%d) ready for prediction in %.4f seconds"
        % (len(to_predict), time.time() - _start)
    )

    all_input_images = []
    all_ground_truths = []
    all_predictions = []

    _start_predict = time.time()
    if isinstance(to_predict, np.ndarray):
        all_predictions = model.predict(to_predict)

    elif isinstance(to_predict, list):
        if batch_size == -1:
            batch_size = get_dynamic_batch_size(model_img_size)
        gen = MultiTargetDataset(
            batch_size,
            model_img_size,
            to_predict,
            notions=notions,
            augment=augment,
            target=target,
        )
        for i, (input_images, ground_truths) in enumerate(gen):
            _start = time.time()
            predictions = np.array(model.predict(input_images))
            # , use_multiprocessing=True, workers=batch_size, max_queue_size=2*batch_size)
            if isinstance(all_input_images, list):
                all_input_images = input_images
            else:
                all_input_images = np.vstack([all_input_images, input_images])
            if isinstance(all_ground_truths, list):
                all_ground_truths = ground_truths
            else:
                all_ground_truths = np.vstack([all_ground_truths, ground_truths])
            if isinstance(all_predictions, list):
                all_predictions = predictions
            else:
                all_predictions = np.hstack([all_predictions, predictions])
            all_image_paths += gen.batch_img_paths
    end = time.time()
    print(
        "%d images predicted in %.4f seconds (%.4f per image)"
        % (
            len(to_predict),
            end - _start_predict,
            (end - _start_predict) / len(to_predict),
        )
    )

    if save:
        if all_image_paths == []:
            all_image_paths = [
                "/tmp/%d_%s.jpg" % (k, prefix) for k in range(len(to_predict))
            ]
        if all_input_images == []:
            all_input_images = to_predict
        save_predictions(
            all_input_images,
            all_predictions,
            all_image_paths,
            all_ground_truths,
            notions,
            notion_indices,
            model_img_size,
            model_name=prefix,
            train=train,
            target=target,
        )

    return all_predictions
