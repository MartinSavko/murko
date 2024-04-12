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

import matplotlib.pyplot as plt
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops
from skimage.transform import resize
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras.models import Model, load_model
from tensorflow.keras.preprocessing import image
from tensorflow.keras.preprocessing.image import (
    save_img,
    load_img,
    img_to_array,
    array_to_img,
)
from tensorflow.keras import regularizers
from tensorflow.keras import layers
from tensorflow import keras
import tensorflow.experimental.numpy as tnp
import tensorflow as tf
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
import seaborn as sns
import simplejpeg
import copy

import scipy.ndimage as ndi

sns.set(color_codes=True)
# from matplotlib import rc
# rc('font', **{'family':'serif','serif':['Palatino']})
# rc('text', usetex=True)
from dataset_loader import MultiTargetDataset, get_dynamic_batch_size, size_differs, get_transposed_img_and_target, get_transformed_img_and_target, get_hierarchical_mask_from_target

try:
    from skimage.morphology.footprints import disk
except BaseException:
    from skimage.morphology.selem import disk

directory = "images_and_labels_augmented"
img_size = (1024, 1360)
model_img_size = (512, 512)
num_classes = 1
batch_size = 8
params = {
    "segmentation": {"loss": "binary_focal_crossentropy", "metrics": "BIoU"},
    "regression": {"loss": "mean_squared_error", "metrics": "mean_absolute_error"},
    "categorical_segmentation": {"loss": "categorical_focal_crossentropy", "metrics": "MeanIoU"},
    "click_segmentation": {"loss": "binary_focal_crossentropy", "metrics": "BIoUm"},
    "click_regression": {
        "loss": "mean_squared_error",
        "metrics": "mean_absolute_error",
    },
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

calibrations = {
    1: np.array([0.00160829, 0.001612]),
    2: np.array([0.00129349, 0.0012945]),
    3: np.array([0.00098891, 0.00098577]),
    4: np.array([0.00075432, 0.00075136]),
    5: np.array([0.00057437, 0.00057291]),
    6: np.array([0.00043897, 0.00043801]),
    7: np.array([0.00033421, 0.00033406]),
    8: np.array([0.00025234, 0.00025507]),
    9: np.array([0.00019332, 0.00019494]),
    10: np.array([0.00015812, 0.00015698]),
}

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


def compare(h1, h2, what="crystal"):
    pylab.figure(1)
    for key in h1:
        if what in key and "loss" not in key:
            pylab.plot(h1[key], label=key)
    pylab.legend()
    pylab.figure(2)
    for key in h2:
        if what in key and "loss" not in key:
            pylab.plot(h2[key], label=key)
    pylab.legend()
    pylab.show()


def plot_history(
    history,
    h=None,
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "capillary",
        "ice",
        "foreground",
    ],
):
    if h is None:
        h = pickle.load(open(history, "rb"), encoding="bytes")
    template = history.replace(".history", "")
    pylab.figure(figsize=(16, 9))
    pylab.title(template)
    for notion in notions:
        key = "val_%s_BIoU_1" % notion
        if key in h:
            pylab.plot(h[key], "o-", label=notion)
        else:
            continue
    pylab.ylim([-0.1, 1.1])
    pylab.grid(True)
    pylab.legend()
    pylab.savefig("%s_metrics.png" % template)


def analyse_histories(
    notions=["crystal", "loop_inside", "loop", "stem", "pin", "foreground"]
):
    histories = (
        glob.glob("*.history")
        + glob.glob("experiments/*.history")
        + glob.glob("backup/*.history")
    )
    metrics_table = {}
    for history in histories:
        print(history)
        h = pickle.load(open(history, "rb"), encoding="bytes")
        plot_history(history, h=h, notions=notions)
        val_metrics = []
        for notion in notions:
            key = "val_%s_BIoU_1" % notion
            if key in h:
                val_metrics.append(h["val_%s_BIoU_1" % notion])
        val_metrics = np.array(val_metrics)
        try:
            best = val_metrics.max(axis=1).T
            best
        except BaseException:
            best = "problem in determining expected metrics"

        line = "%s: %s" % (best, history)
        print(line)
        os.system('echo "%s" >> histories.txt' % line)


def resize_images(images, size, method="bilinear", align_corners=False):
    """See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
        "area": tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, methods[method], align_corners)


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


def get_pixels(
    directory="/nfs/data2/Martin/Research/murko/images_and_labels",
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "capillary",
        "ice",
        "foreground",
    ],
    print_table=True,
):
    masks = glob.glob("%s/*/masks.npy" % directory)
    pixel_counts = dict([(notion, 0) for notion in notions])
    pixel_counts["total"] = 0
    for mask in masks:
        m = np.load(mask)
        for k, notion in enumerate(notions):
            pixel_counts[notion] += m[:, :, k].sum()
        pixel_counts["total"] += np.prod(m.shape[:2])
    if print_table:
        print(
            "total pixels %d (%.3fG)".ljust(15)
            % (pixel_counts["total"], pixel_counts["total"] / 1e9)
        )
        print(
            "total foreground %d (%.3fG, %.4f of all)".ljust(15)
            % (
                pixel_counts["foreground"],
                pixel_counts["foreground"] / 1e9,
                pixel_counts["foreground"] / pixel_counts["total"],
            )
        )
        print()
        print(
            "notion".rjust(15),
            "fraction_label".rjust(15),
            "fraction_total".rjust(15),
            "weight_label".rjust(20),
            "weight_total".rjust(20),
        )
        for key in pixel_counts:
            print(
                key.rjust(15),
                "%.4f".rjust(10) % (pixel_counts[key] / pixel_counts["foreground"]),
                "%.4f".rjust(15) % (pixel_counts[key] / pixel_counts["total"]),
                "%3.1f".zfill(2).rjust(20)
                % (pixel_counts["foreground"] / pixel_counts[key]),
                "%3.1f".zfill(2).rjust(20)
                % (pixel_counts["total"] / pixel_counts[key]),
            )
    return pixel_counts


def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts
            )

            return flops.total_float_ops


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


def generate_click_loss_and_metric_figures(
    click_radius=360e-3, image_shape=(1024, 1360), nclicks=10, ntries=1000, display=True
):
    resize_factor = np.array(image_shape) / np.array((1024, 1360))
    distances = []
    bfcs = []
    bio1 = []
    bio1m = tf.keras.metrics.BinaryIoUm(target_class_ids=[1], threshold=0.5)
    bio0 = []
    bio0m = tf.keras.metrics.BinaryIoUm(target_class_ids=[0], threshold=0.5)
    biob = []
    biobm = tf.keras.metrics.BinaryIoUm(target_class_ids=[0, 1], threshold=0.5)
    concepts = {
        "bfcs": bfcs,
        "bio1": bio1,
        "bio0": bio0,
        "biob": biob,
        "distances": distances,
    }

    for k in range(nclicks):
        click = (
            np.array(image_shape)
            * np.random.rand(
                2,
            )
        ).astype(int)
        cpi_true = click_probability_image(
            click[1],
            click[0],
            image_shape,
            click_radius=click_radius,
            resize_factor=resize_factor,
            scale_click=False,
        )
        cpi_true = np.expand_dims(cpi_true, (0, -1))
        for n in range(ntries // nclicks):
            tclick = (
                np.array(image_shape)
                * np.random.rand(
                    2,
                )
            ).astype(int)
            cpi_pred = click_probability_image(
                tclick[1],
                tclick[0],
                image_shape,
                click_radius=click_radius,
                resize_factor=resize_factor,
                scale_click=False,
            )
            cpi_pred = np.expand_dims(cpi_pred, (0, -1))
            concepts["distances"].append(np.linalg.norm(click - tclick, 2))
            concepts["bfcs"].append(
                tf.keras.losses.binary_focal_crossentropy(cpi_true, cpi_pred)
                .numpy()
                .mean()
            )
            bio1m.reset_state()
            bio1m.update_state(cpi_true, cpi_pred)
            concepts["bio1"].append(bio1m.result().numpy())
            bio0m.reset_state()
            bio0m.update_state(cpi_true, cpi_pred)
            concepts["bio0"].append(bio0m.result().numpy())
            biobm.reset_state()
            biobm.update_state(cpi_true, cpi_pred)
            concepts["biob"].append(biobm.result().numpy())

    for concept in concepts:
        concepts[concept] = np.array(concepts[concept])
    concepts["distances"] /= np.linalg.norm(image_shape, 2)
    concepts["bfcs"] /= concepts["bfcs"].max()
    pylab.figure(figsize=(16, 9))
    pylab.title(
        "image shape %dx%d, click_radius=%.3f"
        % (image_shape[0], image_shape[1], click_radius)
    )
    for concept in ["bfcs", "bio1", "bio0", "biob"]:
        pylab.plot(concepts["distances"], concepts[concept], "o", label=concept)
    pylab.xlabel("distances")
    pylab.ylabel("loss/metrics")
    pylab.savefig(
        "click_metric_cr_%.3f_img_shape_%dx%d.png"
        % (click_radius, image_shape[0], image_shape[1])
    )
    pylab.legend()
    if display:
        pylab.show()
    return concepts


class ClickMetric(tf.keras.metrics.MeanAbsoluteError):
    def __init__(self, name="click_metric", dtype=None):
        super(ClickMetric, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred, sample_weight)


class ClickLoss(tf.keras.losses.MeanSquaredError):
    def call(self, ci_true, ci_pred):
        com_true = tf_center_of_mass(ci_true)
        com_pred = tf_centre_of_mass(ci_pred)

        mse = super().call(com_true, com_pred)
        mse = replacenan(mse)
        bcl = tf.reduce_mean(
            tf.keras.losses.binary_crossentropy(ci_true, ci_pred), axis=(1, 2)
        )
        click_present = tf.reshape(K.max(ci_true, axis=(1, 2)), (-1))
        total = bcl * (1 - click_present) + mse * (click_present)

        return total


def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return np.exp(
        -((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0))
    )


def click_probability_image(
    click_x,
    click_y,
    img_shape,
    zoom=1,
    click_radius=320e-3,
    resize_factor=1.0,
    scale_click=True,
):
    x = np.arange(0, img_shape[1], 1)
    y = np.arange(0, img_shape[0], 1)
    x, y = np.meshgrid(x, y)
    if scale_click:
        mmppx = calibrations[zoom] / resize_factor
    else:
        mmppx = calibrations[1] / resize_factor
    sx = click_radius / mmppx.mean()
    sy = sx
    z = gauss2d(x, y, mx=click_x, my=click_y, sx=sx, sy=sy)
    return z


def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)


def click_loss(ci_true, ci_pred):
    total = tf.keras.losses.mean_squared_error(ci_true, ci_pred)
    return total


def tf_center_of_mass(image_batch, threshold=0.5):
    """https://stackoverflow.com/questions/51724450/finding-centre-of-mass-of-tensor-tensorflow"""
    print(image_batch.shape)
    tf.cast(image_batch >= threshold, tf.float32)
    batch_size, height, width, depth = image_batch.shape
    # Make array of coordinates (each row contains three coordinates)

    ii, jj, kk = tf.meshgrid(
        tf.range(height), tf.range(width), tf.range(depth), indexing="ij"
    )
    coords = tf.stack(
        [tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,)), tf.reshape(kk, (-1))], axis=-1
    )
    coords = tf.cast(coords, tf.float32)
    # Rearrange input into one vector per volume
    volumes_flat = tf.reshape(image_batch, [-1, height * width, 1])
    # Compute total mass for each volume
    total_mass = tf.reduce_sum(volumes_flat, axis=1)
    # Compute centre of mass
    centre_of_mass = tf.reduce_sum(volumes_flat * coords, axis=1) / total_mass

    return centre_of_mass


def click_image_loss(ci_true, ci_pred):
    if K.max(ci_true) == 0:
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    y_true = centre_of_mass(ci_true)
    y_pred = centre_of_mass(ci_pred)
    return tf.keras.losses.mean_squared_error(y_true, y_pred)


def click_mean_absolute_error(ci_true, ci_pred):
    if K.max(ci_pred) < 0.5 and K.max(ci_true) == 0:
        return 0

    y_true = centre_of_mass(ci_true)
    y_pred = centre_of_mass(ci_pred)
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)


def click_batch_loss(click_true_batch, click_pred_image_batch):
    return [
        click_loss(click_true, click_pred_image)
        for click_true, click_pred_image in zip(
            click_true_batch, click_pred_image_batch
        )
    ]


def get_click_from_single_click_image(click_image):
    click_pred = np.zeros((3,))
    m = click_image.max()
    click_pred[:2] = np.array(
        np.unravel_index(np.argmax(click_image), click_image.shape)[:2], dtype="float32"
    )
    click_pred[2] = m
    return click_pred


def get_clicks_from_click_image_batch(click_image_batch):
    input_shape = click_image_batch.shape
    output_shape = (input_shape[0], 3)
    click_pred = np.zeros(output_shape)
    for k, click_image in enumerate(click_image_batch):
        click_pred[k] = get_click_from_single_click_image(click_image)
    return click_pred


def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)


def display_target(target_array):
    normalized_array = (target_array.astype("uint8")) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])

def get_dataset(batch_size, img_size, img_paths, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)


def augment_sample(
    img_path,
    img,
    target,
    user_click,
    do_swap_backgrounds,
    do_flip,
    do_transpose,
    zoom,
    candidate_backgrounds,
    notions,
    zoom_factor,
    shift_factor,
    shear_factor,
):
    if do_swap_backgrounds is True and "background" not in img_path:
        new_background = random.choice(candidate_backgrounds[zoom])
        if size_differs(img.shape[:2], new_background.shape[:2]):
            new_background = resize(new_background, img.shape[:2], anti_aliasing=True)
        img[target[:, :, notions.index("foreground")] == 0] = new_background[
            target[:, :, notions.index("foreground")] == 0
        ]

    if self.augment and do_transpose is True:
        img, target = get_transposed_img_and_target(img, target)

    if self.augment and do_flip is True:
        img, target = get_flipped_img_and_target(img, target)

    if self.augment:
        img, target = get_transformed_img_and_target(
            img,
            target,
            zoom_factor=zoom_factor,
            shift_factor=shift_factor,
            shear_factor=shear_factor,
        )

    return img, target


def get_img_and_target(img_path, img_string="img.jpg", label_string="masks.npy"):
    original_image = load_img(img_path)
    original_size = original_image.size[::-1]
    img = img_to_array(original_image, dtype="float32") / 255.0
    masks_name = img_path.replace(img_string, label_string)
    target = np.load(masks_name)
    return img, target


def get_img(img_path, size=(224, 224)):
    original_image = load_img(img_path)
    img = img_to_array(original_image, dtype="float32") / 255.0
    img = resize(img, size, anti_aliasing=True)
    return img


def load_ground_truth_image(path, target_size):
    ground_truth = np.expand_dims(
        load_img(path, target_size=target_size, color_mode="grayscale"), 2
    )
    if ground_truth.max() > 0:
        ground_truth = np.array(ground_truth / ground_truth.max(), dtype="uint8")
    else:
        ground_truth = np.array(ground_truth, dtype="uint8")
    return ground_truth


def get_cpi_from_user_click(
    user_click,
    img_size,
    resize_factor,
    img_path,
    click_radius=320e-3,
    zoom=1,
    scale_click=False,
):
    if all(np.array(user_click) >= 0):
        try:
            _y = int(user_click[0])
            _x = int(min(user_click[1], img_size[1]))
            if all(np.array((_y, _x)) >= 0):
                cpi = click_probability_image(
                    _x,
                    _y,
                    img_size,
                    click_radius=click_radius,
                    zoom=zoom,
                    resize_factor=resize_factor,
                    scale_click=scale_click,
                )
            else:
                cpi = np.zeros(img_size, dtype="float32")
        except BaseException:
            print(traceback.print_exc())
            os.system("echo %s >> click_generation_problems_new.txt" % img_path)
            return None
    else:
        cpi = np.zeros(img_size, dtype="float32")
    cpi = np.expand_dims(cpi, axis=2)
    return cpi


def get_data_augmentation():
    data_augmentation = keras.Sequential(
        [
            layers.RandomRotation(0.5),
            layers.RandomFlip(),
            layers.RandomZoom(0.2),
        ]
    )
    return data_augmentation


def get_resize_and_rescale(model_img_size):
    resize_and_rescale = keras.Sequential(
        [
            # layers.Resizing(model_img_size, model_img_size),
            layers.Rescaling(1.0 / 255)
        ]
    )
    return resize_and_rescale


def find_number_of_groups(c, g):
    w, r = divmod(c, g)
    if r == 0:
        return g
    else:
        return find_number_of_groups(c, g - 1)


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
    if weight_standardization:
        if convolution_type == "SeparableConv2D":
            x = WSSeparableConv2D(
                filters,
                filter_size,
                padding=padding,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=get_kernel_regularizer(
                    kernel_regularizer, weight_decay
                ),
            )(x)
        elif convolution_type == "Conv2D":
            x = WSConv2D(
                filters,
                filter_size,
                padding=padding,
                use_bias=use_bias,
                kernel_initializer=kernel_initializer,
                kernel_regularizer=get_kernel_regularizer(
                    kernel_regularizer, weight_decay
                ),
            )(x)
    else:
        x = getattr(layers, convolution_type)(
            filters,
            filter_size,
            padding=padding,
            use_bias=use_bias,
            kernel_initializer=kernel_initializer,
            kernel_regularizer=get_kernel_regularizer(kernel_regularizer, weight_decay),
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
        x = layers.Activation(activation=activation)(x)
    else:
        x = get_normalization_layer(
            x,
            normalization_type,
            bn_momentum=bn_momentum,
            bn_epsilon=bn_epsilon,
            gn_groups=gn_groups,
        )
        x = layers.Activation(activation=activation)(x)
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
        x = layers.Dropout(dropout_rate)(x)
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
        x = layers.Concatenate(axis=3)([x, la])
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
    x = layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(x)
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
    **kwargs
):
    x = layers.Concatenate(axis=3)(block_to_upsample[1:])
    x = layers.Conv2DTranspose(
        filters,
        kernel_size=3,
        strides=2,
        padding=padding,
        activation=activation,
        kernel_regularizer=get_kernel_regularizer(kernel_regularizer, weight_decay),
    )(x)
    x = layers.Concatenate(axis=3)([x, skip_connection])
    return x


def get_normalization_layer(
    x, normalization_type, bn_momentum=0.9, bn_epsilon=1.1e-5, gn_groups=16
):
    if normalization_type in ["BN", "BatchNormalization"]:
        x = layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(x)
    elif normalization_type in ["GN", "GroupNormalization"]:
        x = layers.GroupNormalization(
            groups=find_number_of_groups(x.shape[-1], gn_groups)
        )(x)
    elif normalization_type == "BCN":
        x = layers.GroupNormalization(
            groups=find_number_of_groups(x.shape[-1], gn_groups)
        )(x)
        x = layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(x)
    return x

def get_num_segmentation_classes(heads):
    num_segmentation_classes = 0
    for head in heads:
        if head["type"] == "segmentation":
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
        {"name": "crystal", "type": "segmentation"},
        {"name": "loop_inside", "type": "segmentation"},
        {"name": "loop", "type": "segmentation"},
        {"name": "stem", "type": "segmentation"},
        {"name": "pin", "type": "segmentation"},
        {"name": "foreground", "type": "segmentation"},
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

    inputs = layers.Input(shape=(model_img_size) + (input_channels,))

    nfilters_start = nfilters

    if input_dropout > 0.0:
        x = layers.Dropout(dropout_rate=input_dropout)(inputs)
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
    for l, number_of_layers in enumerate(layers_scheme):
        x, block_to_upsample = get_dense_block(
            x, growth_rate, number_of_layers, **boilerplate
        )
        _skips.append(x)
        nfilters += number_of_layers * growth_rate
        x = get_transition_down(x, nfilters, **boilerplate)
        if verbose:
            print("layer:", l, number_of_layers, "shape:", x.shape)
    x, block_to_upsample = get_dense_block(x, growth_rate, bottleneck, **boilerplate)
    if verbose:
        print("bottleneck:", l, number_of_layers, "shape:", x.shape)
    _skips = _skips[::-1]
    extended_layers_scheme = layers_scheme + [bottleneck]
    extended_layers_scheme.reverse()
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

    outputs = []
    regression_neck = None
    num_segmentation_classes = get_num_segmentation_classes(heads)
    for head in heads:
        if head["type"] == "segmentation" or head["type"] == "click_segmentation" or head["name"] == "identity":
            output = layers.Conv2D(
                1,
                1,
                activation="sigmoid",
                padding="same",
                dtype="float32",
                name=head["name"],
            )(x_up)
        elif head["type"] == "categorical_segmentation" and num_segmentation_classes>0:
            output = layers.Conv2D(
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
                # regression_neck = layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype="float32", name=head['name'])(x_up)
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
                regression_neck = layers.Flatten()(regression_neck)
                regression_neck = layers.Dropout(dropout_rate)(regression_neck)
                # regression_neck = get_transition_down(regression_neck, 512, strides=11, pool_size=11)
                # regression_neck = layers.GlobalMaxPool2D()(regression_neck)
                # regression_neck = layers.Flatten()(regression_neck)
            output = layers.Dense(
                3, activation="sigmoid", dtype="float32", name=head["name"]
            )(regression_neck)
        outputs.append(output)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model


def get_tiramisu(
    nfilters=48,
    growth_rate=16,
    layers_scheme=[4, 5, 7, 10, 12],
    bottleneck=15,
    activation="relu",
    convolution_type="Conv2D",
    last_convolution=False,
    dropout_rate=0.2,
    weight_standardization=True,
    model_img_size=(None, None),
    use_bias=False,
    learning_rate=0.001,
    finetune=False,
    finetune_model=None,
    heads=[
        {"name": "crystal", "type": "segmentation"},
        {"name": "loop_inside", "type": "segmentation"},
        {"name": "loop", "type": "segmentation"},
        {"name": "stem", "type": "segmentation"},
        {"name": "pin", "type": "segmentation"},
        {"name": "capillary", "type": "segmentation"},
        {"name": "ice", "type": "segmentation"},
        {"name": "foreground", "type": "segmentation"},
        {"name": "click", "type": "click_segmentation"},
    ],
    name="model",
    normalization_type="GroupNormalization",
    limit_loss=True,
    weight_decay=1.0e-4,
):
    print("get_tiramisu heads", heads)
    model = get_uncompiled_tiramisu(
        nfilters=nfilters,
        growth_rate=growth_rate,
        layers_scheme=layers_scheme,
        bottleneck=bottleneck,
        activation=activation,
        convolution_type=convolution_type,
        last_convolution=last_convolution,
        dropout_rate=dropout_rate,
        weight_standardization=weight_standardization,
        model_img_size=model_img_size,
        heads=heads,
        name=name,
        normalization_type=normalization_type,
        weight_decay=weight_decay,
    )
    if finetune and finetune_model is not None:
        print("loading weights to finetune")
        model.load_weights(finetune_model)
    else:
        print("not finetune")
    losses = {}
    metrics = {}
    num_segmentation_classes = get_num_segmentation_classes(heads)
    for head in heads:
        losses[head["name"]] = params[head["type"]]["loss"]
        print('head name and type', head["name"], head["type"])
        if params[head["type"]]["metrics"] == "BIoU":
            metrics[head["name"]] = [
                tf.keras.metrics.BinaryIoU(
                    target_class_ids=[1], threshold=0.5, name="BIoU_1"
                ),
                tf.keras.metrics.BinaryIoU(
                    target_class_ids=[0], threshold=0.5, name="BIoU_0"
                ),
                tf.keras.metrics.BinaryIoU(
                    target_class_ids=[0, 1], threshold=0.5, name="BIoU_both"
                ),
            ]
        elif params[head["type"]]["metrics"] == "BIoUm":
            metrics[head["name"]] = [
                tf.keras.metrics.BinaryIoUm(
                    target_class_ids=[1], threshold=0.5, name="BIoUm_1"
                ),
                tf.keras.metrics.BinaryIoUm(
                    target_class_ids=[0], threshold=0.5, name="BIoUm_0"
                ),
                tf.keras.metrics.BinaryIoUm(
                    target_class_ids=[0, 1], threshold=0.5, name="BIoUm_both"
                ),
            ]
        elif params[head["type"]]["metrics"] == "mean_absolute_error":
            metrics[head["name"]] = tf.keras.metrics.MeanAbsoluteError(name="MAE")
        elif head["type"] == "categorical_segmentation":
            metrics[head["name"]] = getattr(tf.keras.metrics, params[head["type"]]["metrics"])(num_segmentation_classes+1)
            
            #, sparse_y_true=True, sparse_y_pred=True)
            #losses[head["name"]] = tf.keras.losses.BinaryFocalCrossentropy(name="hierarchy_loss", from_logits=True)
            #getattr(tf.keras.losses, params[head["type"]]["loss"])(from_logits=True)
        else:
            metrics[head["name"]] = getattr(tf.keras.metrics, params[head["type"]]["metrics"])()
            
    print("losses", len(losses), losses)
    print("metrics", len(metrics), metrics)
    loss_weights = {}
    for head in heads:
        if head["name"] in loss_weights_from_stats:
            lw = loss_weights_from_stats[head["name"]]
            if limit_loss:
                if lw > loss_weights_from_stats["crystal"]:
                    lw = loss_weights_from_stats["crystal"]
        else:
            lw = 1
        loss_weights[head["name"]] = lw

    print("loss weights", loss_weights)
    lrs = learning_rate
    # lrs = tf.keras.optimizers.schedules.ExponentialDecay(lrs, decay_steps=1e4, decay_rate=0.96, minimum_value=1e-7, staircase=True)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lrs)
    # optimizer = tf.keras.optimizers.Adam(learning_rate=lrs)
    if finetune:
        for l in model.layers[: -len(heads)]:
            l.trainable = False

    model.compile(
        optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics
    )

    print("model.losses", len(model.losses), model.losses)
    print("model.metrics", len(model.metrics), model.metrics)
    return model


def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))


def path_to_target(path):
    img = img_to_array(load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8")
    return img


def get_paths(directory="images_and_labels", seed=1337):
    input_img_paths = glob.glob(os.path.join(directory, "*/img.jpg"))
    target_img_paths = [
        item.replace("img.jpg", "foreground.png") for item in input_img_paths
    ]
    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)
    return input_img_paths, target_img_paths


def get_training_dataset(seed=1337, num_val_samples=150):
    input_img_paths, target_img_paths = get_paths(seed=seed)
    train_paths = input_img_paths[:-num_val_samples]
    train_target_img_paths = target_img_paths[:-num_val_samples]
    return train_paths, train_target_img_paths


def get_validation_dataset(seed=1337, num_val_samples=150):
    input_img_paths, target_img_paths = get_paths(seed=seed)
    val_paths = input_img_paths[-num_val_samples:]
    val_target_img_paths = target_img_paths[-num_val_samples:]
    return val_paths, val_target_img_paths


def get_family(name):
    fname = os.path.realpath(name)
    search_string = ".*/double_clicks_(.*)_double_click.*|.*/(.*)_manual_omega.*|.*/(.*)_color_zoom.*|.*/(.*)_auto_omega.*"
    match = re.findall(search_string, fname)
    print("match", match)
    if match:
        for item in match[0]:
            if item != "":
                return item
    else:
        return os.path.basename(os.path.dirname(fname))


def get_sample_families(directory="images_and_labels", subset_designation="*"):
    search_string = "{directory:s}/double_clicks_(.*)_double_click.*|{directory:s}/(.*)_manual_omega.*|{directory:s}/(.*)_color_zoom.*|{directory:s}/(.*)_auto_omega.*".format(
        directory=directory
    )
    individuals = glob.glob("%s/%s" % (directory, subset_designation))
    sample_families = {}
    for individual in individuals:
        matches = re.findall(search_string, individual)
        individual = individual.replace("%s/" % directory, "")
        if matches:
            for match in matches[0]:
                if match != "":
                    if match in sample_families:
                        sample_families[match].append(individual)
                    else:
                        sample_families[match] = [individual]
        else:
            sample_families[individual] = [individual]
    return sample_families


def get_paths_for_families(families_subset_list, sample_families, directory):
    paths = []
    for family in families_subset_list:
        for individual in sample_families[family]:
            paths.append(os.path.join(directory, individual, "img.jpg"))
    return paths


def get_training_and_validation_datasets(
    directory="images_and_labels", seed=12345, split=0.2
):
    sample_families = get_sample_families(directory=directory)
    sample_families_names = sorted(sample_families.keys())
    random.Random(seed).shuffle(sample_families_names)
    total = len(sample_families_names)

    train = int((1 - split) * total)
    train_families = sample_families_names[:train]
    valid_families = sample_families_names[train:]
    print("total %d" % total)
    print("train", train)
    print("train_families: %d" % len(train_families))
    print("valid_families: %d" % len(valid_families))

    train_paths = get_paths_for_families(train_families, sample_families, directory)
    random.Random(seed).shuffle(train_paths)
    val_paths = get_paths_for_families(valid_families, sample_families, directory)
    random.Random(seed).shuffle(val_paths)

    return train_paths, val_paths


def get_training_and_validation_datasets_for_clicks(
    basedir="./",
    seed=1,
    background_percent=10,
    train_images=10000,
    valid_images=2500,
    forbidden=[],
):
    backgrounds = glob.glob(
        os.path.join(basedir, "shapes_of_background/*.jpg")
    ) + glob.glob(os.path.join(basedir, "Backgrounds/*.jpg"))
    random.Random(seed).shuffle(backgrounds)
    train_paths = glob.glob(
        os.path.join(basedir, "unique_shapes_of_clicks/*.jpg")
    )  # + glob.glob('images_and_labels_augmented/*/img.jpg')
    random.Random(seed).shuffle(train_paths)
    train_paths = train_paths[:train_images]
    backgrounds = backgrounds[: int(len(train_paths) / background_percent)]
    train_paths += backgrounds
    random.Random(seed).shuffle(train_paths)
    val_paths = train_paths[-valid_images:]
    if len(train_paths) - valid_images < train_images:
        train_paths = train_paths[:train_images]
    else:
        train_paths = train_paths[:-valid_images]
    if len(forbidden) > 0:
        train_paths = [item for item in train_paths if item not in forbidden]
    return train_paths, val_paths


def segment_multihead(
    base="/nfs/data2/Martin/Research/murko",
    epochs=25,
    patience=3,
    mixed_precision=False,
    name="start",
    source_weights=None,
    batch_size=16,
    model_img_size=(512, 512),
    network="fcdn56",
    convolution_type="SeparableConv2D",
    heads=[
        {"name": "crystal", "type": "segmentation"},
        {"name": "loop_inside", "type": "segmentation"},
        {"name": "loop", "type": "segmentation"},
        {"name": "stem", "type": "segmentation"},
        {"name": "pin", "type": "segmentation"},
        {"name": "capillary", "type": "segmentation"},
        {"name": "ice", "type": "segmentation"},
        {"name": "foreground", "type": "segmentation"},
        {"name": "click", "type": "click_segmentation"},
    ],
    last_convolution=False,
    augment=True,
    train_images=-1,
    valid_images=1000,
    scale_click=False,
    click_radius=320e-3,
    learning_rate=0.001,
    pixel_budget=768 * 992,
    normalization_type="GroupNormalization",
    validation_scale=0.4,
    dynamic_batch_size=True,
    finetune=False,
    seed=12345,
    artificial_size_increase=1,
    include_plate_images=False,
    include_capillary_images=False,
    dropout_rate=0.2,
    weight_standardization=True,
    limit_loss=True,
    weight_decay=1.0e-4,
    activation="relu",
    train_dev_split=0.2,
    val_model_img_size=(256, 320),
):
    if mixed_precision:
        print("setting mixed_precision")
        keras.mixed_precision.set_global_policy("mixed_float16")

    for gpu in tf.config.list_physical_devices("GPU"):
        print("setting memory_growth on", gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    notions = [head["name"] for head in heads]
    distinguished_name = "%s_%s" % (network, name)
    model_name = os.path.join(base, "%s.h5" % distinguished_name)
    history_name = os.path.join(base, "%s.history" % distinguished_name)
    png_name = os.path.join(base, "%s_losses.png" % distinguished_name)
    checkpoint_filepath = "%s_{batch:06d}_{loss:.4f}.h5" % distinguished_name
    # segment_train_paths, segment_val_paths = get_training_and_validation_datasets()
    # print('training on %d samples, validating on %d samples' % ( len(train_paths), len(val_paths)))
    # data genrators
    train_paths, val_paths = get_training_and_validation_datasets(
        directory=os.path.join(base, "images_and_labels"), split=train_dev_split
    )
    if include_plate_images:
        train_paths_plate, val_paths_plate = get_training_and_validation_datasets(
            directory=os.path.join(base, "images_and_labels_plate"), split=0
        )
        # val_paths += val_paths_plate
        train_paths += train_paths_plate
    if include_capillary_images:
        (
            train_paths_capillary,
            val_paths_capillary,
        ) = get_training_and_validation_datasets(
            directory=os.path.join(base, "images_and_labels_capillary"), split=0
        )
        # val_paths += val_paths_plate
        train_paths += train_paths_capillary
        val_paths += val_paths_capillary
    full_size = len(train_paths)
    if train_images != -1:
        train_paths = train_paths[:train_images]
        factor = full_size // len(train_paths)
        train_paths = train_paths * (factor + 1)
        random.Random(seed).shuffle(train_paths)
        train_paths = train_paths[:full_size]

    # train_paths, val_paths = get_training_and_validation_datasets_for_clicks(basedir='/dev/shm', train_images=train_images, valid_images=valid_images, forbidden=[])
    print("\ntotal number of samples %d" % len(train_paths + val_paths))
    print(
        "training on %d samples, validating on %d samples\n"
        % (len(train_paths), len(val_paths))
    )
    # train_gen = CrystalClickDataset(batch_size, model_img_size, train_paths, augment=augment, scale_click=scale_click, click_radius=click_radius, dynamic_batch_size=dynamic_batch_size, shuffle_at_0=True)
    print("notions in segment_multihead", notions)
    train_gen = MultiTargetDataset(
        batch_size,
        model_img_size,
        train_paths,
        notions=notions,
        augment=augment,
        transform=True,
        flip=True,
        transpose=True,
        scale_click=scale_click,
        click_radius=click_radius,
        dynamic_batch_size=dynamic_batch_size,
        pixel_budget=pixel_budget,
        artificial_size_increase=artificial_size_increase,
        shuffle_at_0=True,
        black_and_white=True,
    )
    if val_model_img_size is None:
        val_model_img_size = get_img_size_as_scale_of_pixel_budget(validation_scale)
    val_batch_size = get_dynamic_batch_size(val_model_img_size)
    print("validation model_img_size will be", val_model_img_size)
    # val_gen = CrystalClickDataset(val_batch_size, val_model_img_size, val_paths, augment=False, scale_click=scale_click, click_radius=click_radius, dynamic_batch_size=False)
    val_gen = MultiTargetDataset(
        val_batch_size,
        val_model_img_size,
        val_paths,
        augment=False,
        transform=False,
        notions=notions,
        pixel_budget=pixel_budget,
    )
    # callbacks
    checkpointer = keras.callbacks.ModelCheckpoint(
        model_name, verbose=1, monitor="val_loss", save_best_only=True, mode="min"
    )
    # checkpointer2 = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, monitor='loss', save_freq=2000, save_best_only=False, mode='min')
    nanterminator = keras.callbacks.TerminateOnNaN()
    # tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(os.path.realpath('./'), '%s_logs' % model_name.replace('.h5', '')), update_freq='epoch', write_steps_per_second=True)
    # earlystopper = keras.callbacks.EarlyStopping(patience=patience, verbose=1)
    lrreducer = (
        keras.callbacks.ReduceLROnPlateau(
            factor=0.75,
            monitor="val_loss",
            patience=patience,
            cooldown=1,
            min_lr=1e-6,
            verbose=1,
        ),
    )
    callbacks = [checkpointer, nanterminator, lrreducer]
    network_parameters = networks[network]

    if os.path.isdir(model_name) or os.path.isfile(model_name):
        print("model exists, loading weights ...")
        # model = keras.models.load_model(model_name)
        model = get_tiramisu(
            convolution_type=convolution_type,
            model_img_size=(None, None),
            heads=heads,
            last_convolution=last_convolution,
            name=network,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            weight_standardization=weight_standardization,
            normalization_type=normalization_type,
            finetune=finetune,
            finetune_model=model_name,
            limit_loss=limit_loss,
            weight_decay=weight_decay,
            activation=activation,
            **network_parameters
        )
        if not finetune:
            model.load_weights(model_name)
        history_name = history_name.replace(".history", "_next_superepoch.history")
        png_name = png_name.replace(".png", "_next_superepoch.png")
    else:
        print(model_name, "does not exist")
        # custom_objects = {"click_loss": click_loss, "ClickMetric": ClickMetric}
        # with keras.utils.custom_object_scope(custom_objects):
        model = get_tiramisu(
            convolution_type=convolution_type,
            model_img_size=(None, None),
            heads=heads,
            last_convolution=last_convolution,
            name=network,
            learning_rate=learning_rate,
            dropout_rate=dropout_rate,
            weight_standardization=weight_standardization,
            normalization_type=normalization_type,
            limit_loss=limit_loss,
            weight_decay=weight_decay,
            activation=activation,
            **network_parameters
        )
        # sys.exit()
    print(model.summary())

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        max_queue_size=64,
        use_multiprocessing=True,
        workers=32,
    )

    f = open(history_name, "wb")
    pickle.dump(history.history, f)
    f.close()

    epochs = range(1, len(history.history["loss"]) + 1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure(figsize=(16, 9))
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(png_name)
    plot_history(png_name.replace("_losses.png", ""), history.history)


def segment(base="/nfs/data2/Martin/Research/murko", epochs=25, patience=5):
    keras.mixed_precision.set_global_policy("mixed_float16")
    date = "start"
    date_next = "2021-12-13_tiramisu_SeparableConv2D"
    model_and_weights = os.path.join(base, "sample_segmentation_%s.h5" % date)
    next_model_and_weights = os.path.join(
        base, "sample__segmentation_%s.h5" % date_next
    )
    model_train_history = os.path.join(
        base, "sample_segmentation_%s_history.pickle" % date
    )
    # model = get_other_model(model_img_size, num_classes)
    # Split our img paths into a training and a validation set
    # train_paths, train_target_img_paths = get_training_dataset()
    # val_paths, val_target_img_paths = get_validation_dataset()
    train_paths, val_paths = get_training_and_validation_datasets()
    print(
        "training on %d samples, validating on %d samples"
        % (len(train_paths), len(val_paths))
    )

    # Instantiate data Sequences for each split
    train_gen = SampleSegmentationDataset(
        batch_size, model_img_size, train_paths, augment=True
    )
    val_gen = SampleSegmentationDataset(
        batch_size, model_img_size, val_paths, augment=False
    )

    model = get_tiramisu(convolution_type="SeparableConv2D")

    # new_date = '2021-11-12_zoom123'
    checkpointer = keras.callbacks.ModelCheckpoint(
        next_model_and_weights, verbose=1, mode="min", save_best_only=True
    )
    earlystopper = keras.callbacks.EarlyStopping(patience=patience, verbose=1)
    callbacks = [checkpointer, earlystopper]

    # Train the model, doing validation at the end of each epoch.
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    # model = get_tiramisu(convolution_type='SeparableConv2D')
    # if os.path.isfile(model_and_weights):
    # model.load_weights(model_and_weights)
    # history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    history = model.fit(
        train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks
    )
    f = open(model_train_history, "wb")
    pickle.dump(history.history, f)
    f.close()
    # Generate predictions for all images in the validation set
    # val_gen = SampleSegmentationDataset(batch_size, img_size, val_paths, val_target_img_paths)
    _start = time.time()
    val_preds = model.predict(val_gen)
    print(
        "predicting %d examples took %.4f seconds"
        % (len(val_preds), time.time() - _start)
    )
    epochs = range(1, len(history.history["loss"]) + 1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure()
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig("sample_foreground_segmentation_%s.png" % date)
    plt.show()


def efficient_resize(img, new_size, anti_aliasing=True):
    return (
        img_to_array(array_to_img(img).resize(new_size[::-1]), dtype="float32") / 255.0
    )


def get_notion_description(mask, min_size=32):
    present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, properties = [
        np.nan
    ] * 12
    present = 0

    if np.any(mask):
        labeled_image = mask.astype("uint8")
        properties = regionprops(labeled_image)[0]

        if properties.convex_area > min_size:
            present = 1
            area = properties.convex_area
        else:
            present = 0
        bbox = properties.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        r, c = properties.centroid
        c_max = bbox[3]
        r_max = ndi.center_of_mass(labeled_image[:, c_max - 5 : c_max])[0]
        c_min = bbox[1]
        r_min = ndi.center_of_mass(labeled_image[:, c_min : c_min + 5])[0]

    return present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, properties


def get_notion_mask_from_predictions(
    predictions,
    notion,
    k=0,
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    min_size=32,
):
    notion_mask = np.zeros(predictions[0].shape[1:3], dtype=bool)

    if isinstance(notion, list):
        for n in notion:
            index = notion_indices[n]
            noti_pred = predictions[index][k, :, :, 0] > threshold
            noti_pred = remove_small_objects(noti_pred, min_size=min_size)
            notion_mask = np.logical_or(notion_mask, noti_pred)

    elif isinstance(notion, str):
        index = notion_indices[notion]
        notion_mask = predictions[index][k, :, :, 0] > threshold
        notion_mask = remove_small_objects(notion_mask, min_size=min_size)
    return notion_mask


def get_notion_mask_from_masks(
    masks,
    notion,
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    min_size=32,
):
    notion_mask = np.zeros(masks.shape[:2], dtype=bool)

    if isinstance(notion, list):
        for n in notion:
            index = notion_indices[n]
            noti_mask = masks[:, :, index]
            noti_mask = remove_small_objects(noti_mask > 0, min_size=min_size)
            notion_mask = np.logical_or(notion_mask, noti_mask)
    elif isinstance(notion, str):
        index = notion_indices[notion]
        notion_mask = masks[:, :, index]
        notion_mask = remove_small_objects(notion_mask > 0, min_size=min_size)
    return notion_mask


def get_notion_prediction(
    predictions,
    notion,
    k=0,
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    min_size=32,
):
    if isinstance(predictions, list):
        notion_mask = get_notion_mask_from_predictions(
            predictions,
            notion,
            k=k,
            notion_indices=notion_indices,
            threshold=threshold,
            min_size=min_size,
        )
    elif isinstance(predictions, np.ndarray) and len(predictions.shape) == 3:
        notion_mask = get_notion_mask_from_masks(
            predictions, notion, notion_indices=notion_indices, min_size=min_size
        )

    (
        present,
        r,
        c,
        h,
        w,
        r_max,
        c_max,
        r_min,
        c_min,
        bbox,
        area,
        properties,
    ) = get_notion_description(notion_mask, min_size=min_size)

    if not isinstance(properties, float):
        if notion == "foreground" or isinstance(notion, list):
            notion_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = properties.filled_image
        else:
            notion_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = properties.convex_image

    return (
        present,
        r,
        c,
        h,
        w,
        r_max,
        c_max,
        r_min,
        c_min,
        bbox,
        area,
        notion_mask.astype("uint8"),
    )


def get_most_likely_click(predictions, k=0, verbose=False, min_size=32):
    _start = time.time()
    gmlc = False
    most_likely_click = -1, -1

    for notion in ["crystal", "loop_inside", "loop"]:
        notion_prediction = get_notion_prediction(
            predictions, notion, k=k, min_size=min_size
        )
        if notion_prediction[0] == 1:
            most_likely_click = (
                notion_prediction[1] / notion_prediction[-1].shape[0],
                notion_prediction[2] / notion_prediction[-1].shape[1],
            )
            if verbose:
                print("%s found!" % notion)
            gmlc = True
            break
    if gmlc is False:
        foreground = get_notion_prediction(
            predictions, "foreground", k=k, min_size=min_size
        )
        if foreground[0] == 1:
            most_likely_click = (
                foreground[5] / foreground[-1].shape[0],
                foreground[6] / foreground[-1].shape[1],
            )
    if verbose:
        print("most likely click determined in %.4f seconds" % (time.time() - _start))
    return most_likely_click


def get_bbox_from_description(description, notions=[["crystal", "loop"], "foreground"]):
    shape = description["hierarchical_mask"].shape
    for notion in notions:
        notion_description = description[get_notion_string(["crystal", "loop"])]
        present = notion_description["present"]
        if present:
            r = notion_description["r"]
            c = notion_description["c"]
            h = notion_description["h"]
            w = notion_description["w"]
            r /= shape[0]
            c /= shape[1]
            h /= shape[0]
            w /= shape[1]
            break
        else:
            r, c, h, w = 4 * [np.nan]
    return present, r, c, h, w


def get_notion_string(notion):
    if isinstance(notion, list):
        notion_string = ",".join(notion)
    else:
        notion_string = notion
    return notion_string


def get_loop_bbox(predictions, k=0, min_size=32):
    (
        loop_present,
        r,
        c,
        h,
        w,
        r_max,
        c_max,
        r_min,
        c_min,
        bbox,
        area,
        notion_prediction,
    ) = get_notion_prediction(
        predictions, ["crystal", "loop_inside", "loop"], k=k, min_size=min_size
    )
    shape = predictions[0].shape[1:3]
    if bbox is not np.nan:
        r = bbox[0] + h / 2
        c = bbox[1] + w / 2
    r /= shape[0]
    c /= shape[1]
    h /= shape[0]
    w /= shape[1]
    return loop_present, r, c, h, w


def get_raw_projections(predictions, notion="foreground", threshold=0.5, min_size=32):
    raw_projections = []
    for k in range(len(predictions[0])):
        (
            present,
            r,
            c,
            h,
            w,
            r_max,
            c_max,
            r_min,
            c_min,
            bbox,
            area,
            notion_mask,
        ) = get_notion_prediction(predictions, notion, k=k, min_size=min_size)
        raw_projections.append((present, (r, c, h, w), notion_mask))
    return raw_projections


def get_descriptions(
    predictions,
    notions=[
        "foreground",
        "crystal",
        "loop_inside",
        "loop",
        ["crystal", "loop"],
        ["crystal", "loop", "stem"],
    ],
    threshold=0.5,
    min_size=32,
    original_image_shape=(1200, 1600),
):
    descriptions = []
    foreground = get_notion_string("foreground")
    crystal_loop = get_notion_string(["crystal", "loop"])
    possible = get_notion_string(["crystal", "loop", "stem"])
    prediction_shape = np.array(predictions[0].shape[1:3])
    original_shape = np.array(original_image_shape[:2])
    for k in range(len(predictions[0])):
        description = {}
        description["original_shape"] = original_shape
        description["prediction_shape"] = prediction_shape
        description["hierarchical_mask"] = get_hierarchical_mask_from_predictions(
            predictions, k=k
        )
        for notion in notions:
            (
                present,
                r,
                c,
                h,
                w,
                r_max,
                c_max,
                r_min,
                c_min,
                bbox,
                area,
                notion_mask,
            ) = get_notion_prediction(predictions, notion, k=k, min_size=min_size)
            if present:
                epo, epi, epooa, epioa, pa = get_extreme_point(notion_mask)
            else:
                epo, epi, epooa, epioa = 4 * [(-1, -1)]
                pa = np.nan
            # description[get_notion_string(notion)] = (present, (r, c, h, w, area), (epo, epi, epooa, epioa, pa), notion_mask)
            description[get_notion_string(notion)] = {
                "present": present,
                "r": r,
                "c": c,
                "h": h,
                "w": w,
                "area": area,
                "epo": epo,
                "epi": epi,
                "epooa": epooa,
                "epioa": epioa,
                "pa": pa,
                "notion_mask": notion_mask,
            }

        epo_cil, epi_cil, epooa_cil, epioa_cil, pa_cil = get_extreme_point(
            description[crystal_loop]["notion_mask"], pa=description[possible]["pa"]
        )
        description["present"] = description[foreground]["present"]
        description["most_likely_click"] = get_most_likely_click_from_description(
            description
        )
        description["aoi_bbox"] = get_bbox_from_description(
            description, notions=[["crystal", "loop"], "foreground"]
        )
        description["crystal_bbox"] = get_bbox_from_description(
            description, notions=["crystal"]
        )
        description["extreme"] = description[foreground]["epo"] / prediction_shape
        description["end_likely"] = (
            description[crystal_loop]["epooa"] / prediction_shape
        )
        description["start_likely"] = (
            epioa_cil / prediction_shape
        )  # description[crystal_loop]['epioa']
        description["start_possible"] = (
            description[possible]["epioa"] / prediction_shape
        )
        descriptions.append(description)
    return descriptions


def get_most_likely_click_from_description(description, verbose=False):
    _start = time.time()
    gmlc = False
    most_likely_click = -1, -1
    shape = np.array(description["hierarchical_mask"].shape)
    for notion in ["crystal", "loop_inside", "loop"]:
        notion_description = description[get_notion_string(notion)]
        if notion_description["present"]:
            r = notion_description["r"]
            c = notion_description["c"]
            most_likely_click = np.array((r, c)) / shape
            if verbose:
                print("%s found!" % notion)
            gmlc = True
            break
    if gmlc is False:
        notion_description = description["foreground"]
        if notion_description["present"]:
            epo = notion_description["epo"]
            most_likely_click = np.array(epo) / shape
    if verbose:
        print("most likely click determined in %.4f seconds" % (time.time() - _start))
    return most_likely_click


def principal_axes(array, verbose=False):
    # https://github.com/pierrepo/principal_axes/blob/master/principal_axes.py
    _start = time.time()
    if array.shape[1] != 3:
        xyz = np.argwhere(array == 1)
    else:
        xyz = array[:, :]

    coord = np.array(xyz, float)
    center = np.mean(coord, 0)
    coord = coord - center
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)
    order = np.argsort(e_values)[::-1]
    eigenvalues = np.array(e_values[order])
    eigenvectors = np.array(e_vectors[:, order])
    _end = time.time()
    if verbose:
        print("principal axes")
        print("intertia tensor")
        print(inertia)
        print("eigenvalues")
        print(eigenvalues)
        print("eigenvectors")
        print(eigenvectors)
        print("principal_axes calculated in %.4f seconds" % (_end - _start))
        print()
    return inertia, eigenvalues, eigenvectors, center


def get_extreme_point(
    projection, pa=None, orientation="horizontal", extreme_direction=-1
):
    try:
        xyz = np.argwhere(projection != 0)
        if pa is None:
            pa = principal_axes(projection)

        S = pa[-2]
        center = pa[-1]

        xyz_0 = xyz - center

        xyz_S = np.dot(xyz_0, S)
        xyz_S_on_axis = xyz_S[np.isclose(xyz_S[:, 1], 0, atol=5)]

        mino = xyz_S[np.argmin(xyz_S[:, 0])]
        try:
            mino_on_axis = xyz_S_on_axis[np.argmin(xyz_S_on_axis[:, 0])]
        except BaseException:
            print(traceback.print_exc())
            mino_on_axis = copy.copy(mino)
        maxo = xyz_S[np.argmax(xyz_S[:, 0])]
        try:
            maxo_on_axis = xyz_S_on_axis[np.argmax(xyz_S_on_axis[:, 0])]
        except BaseException:
            print(traceback.print_exc())
            maxo_on_axis = copy.copy(maxo)

        mino_0_s = np.dot(mino, np.linalg.inv(S)) + center
        maxo_0_s = np.dot(maxo, np.linalg.inv(S)) + center

        mino_0_s_on_axis = np.dot(mino_on_axis, np.linalg.inv(S)) + center
        maxo_0_s_on_axis = np.dot(maxo_on_axis, np.linalg.inv(S)) + center

        if orientation == "horizontal":
            if extreme_direction * mino_0_s[1] > extreme_direction * maxo_0_s[1]:
                extreme_point_out = mino_0_s
                extreme_point_out_on_axis = mino_0_s_on_axis
                extreme_point_ini = maxo_0_s
                extreme_point_ini_on_axis = maxo_0_s_on_axis
            else:
                extreme_point_out = maxo_0_s
                extreme_point_out_on_axis = maxo_0_s_on_axis
                extreme_point_ini = mino_0_s
                extreme_point_ini_on_axis = mino_0_s_on_axis
        else:
            if extreme_direction * mino_0_s[0] > extreme_direction * maxo_0_s[0]:
                extreme_point_out = mino_0_s
                extreme_point_out_on_axis = mino_0_s_on_axis
                extreme_point_ini = maxo_0_s
                extreme_point_ini_on_axis = maxo_0_s_on_axis
            else:
                extreme_point_out = maxo_0_s
                extreme_point_out_on_axis = maxo_0_s_on_axis
                extreme_point_ini = mino_0_s
                extreme_point_ini_on_axis = mino_0_s_on_axis
    except BaseException:
        print(traceback.print_exc())
        (
            extreme_point_out,
            extreme_point_ini,
            extreme_point_out_on_axis,
            extreme_point_ini_on_axis,
        ) = [[-1, -1]] * 4
    return (
        extreme_point_out,
        extreme_point_ini,
        extreme_point_out_on_axis,
        extreme_point_ini_on_axis,
        pa,
    )


def get_predictions(request_arguments, host="localhost", port=89019, verbose=False):
    start = time.time()
    context = zmq.Context()
    if verbose:
        print("Connecting to server ...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://%s:%d" % (host, port))
    socket.send(pickle.dumps(request_arguments))
    raw_predictions = socket.recv()
    predictions = pickle.loads(raw_predictions)
    if verbose:
        print("Received predictions in %.4f seconds" % (time.time() - start))
    return predictions


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
        layer.name for layer in model.layers[-10:] if isinstance(layer, layers.Conv2D)
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


def get_hierarchical_mask_from_prediction(
    prediction,
    notions=["crystal", "loop_inside", "loop", "stem", "pin", "foreground"],
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    notion_values={
        "crystal": 6,
        "loop_inside": 5,
        "loop": 4,
        "stem": 3,
        "pin": 2,
        "foreground": 1,
    },
    min_size=32,
    massage=True,
):
    hierarchical_mask = np.zeros(prediction.shape[:2])
    for notion in notions:
        notion_value = notion_values[notion]
        l = notion_indices[notion]
        mask = prediction[:, :, l] > threshold
        if massage:
            if notion in ["crystal", "loop", "loop_inside", "stem", "pin"]:
                massager = "convex"
            else:
                massager = "filled"
            mask = massage_mask(mask, min_size=min_size, massager=massager)
        if np.any(mask):
            hierarchical_mask[mask == 1] = notion_value
    return hierarchical_mask


def get_hierarchical_mask_from_kth_prediction(predictions, k):
    prediction = get_kth_prediction_from_predictions(predictions, k)
    hierarchical_mask = get_hierarchical_mask_from_prediction(prediction)
    return hierarchical_mask


def get_hierarchical_mask_from_predictions(
    predictions,
    k=0,
    notions=["crystal", "loop_inside", "loop", "stem", "pin", "foreground"],
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    notion_values={
        "crystal": 6,
        "loop_inside": 5,
        "loop": 4,
        "stem": 3,
        "pin": 2,
        "foreground": 1,
    },
    min_size=32,
    massage=True,
):
    hierarchical_mask = np.zeros(predictions[0].shape[1:3], dtype=np.uint8)
    for notion in notions[::-1]:
        notion_value = notion_values[notion]
        l = notion_indices[notion]
        mask = predictions[l][k, :, :, 0] > threshold
        if massage:
            if notion in ["crystal", "loop", "loop_inside", "stem", "pin"]:
                massager = "convex"
            else:
                massager = "filled"
            mask = massage_mask(mask, min_size=min_size, massager=massager)
        if np.any(mask):
            hierarchical_mask[mask == 1] = notion_value
    return hierarchical_mask


def get_kth_prediction_from_predictions(predictions, k):
    prediction = np.zeros(predictions.shape[1:3] + predictions.shape[0], dtype=np.uint8)
    for n, notion in enumerate(predictions):
        prediction[:, :, n] = predictions[n][k][:, :, 0]
    return prediction


def massage_mask(mask, min_size=32, massager="convex"):
    mask = remove_small_objects(mask, min_size=min_size)
    if not np.any(mask):
        return mask
    labeled_image = mask.astype("uint8")
    properties = regionprops(labeled_image)[0]
    bbox = properties.bbox
    if massager == "convex":
        mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = properties.convex_image
    elif massager == "filled":
        mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = properties.filled_image
    return mask


def plot_analysis(
    input_images, analysis, figsize=(16, 9), model_name="default", image_paths=None
):
    _start = time.time()
    descriptions = analysis["descriptions"]

    for k, input_image in enumerate(input_images):
        hierarchical_mask = descriptions[k]["hierarchical_mask"]

        if image_paths is None or image_paths != []:
            name = os.path.basename(image_paths[k])
            prefix = name[:-4]
            directory = os.path.dirname(image_paths[k])
        else:
            name = "%.1f" % time.time()
            prefix = "test"
            directory = "/tmp"

        print("name, prefix, directory", name, prefix, directory)
        template = "%s_%s_model_img_size_%dx%d" % (
            prefix,
            name,
            hierarchical_mask.shape[0],
            hierarchical_mask.shape[1],
        )
        print("template", template)
        prediction_img_path = os.path.join(
            directory, "%s_hierarchical_mask_high_contrast_predicted.png" % (template)
        )
        # save_img(prediction_img_path, np.expand_dims(hierarchical_mask, axis=2), scale=True)

        # predicted_masks_name = os.path.join(directory, '%s.npy' % template)
        # np.save(predicted_masks_name, predicted_masks)

        fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(figsize)
        fig.suptitle(name)
        axes[0].set_title(
            "input image with predicted click and loop bounding box (if any)"
        )
        axes[0].imshow(input_image)
        axes[1].set_title("raw segmentation result with most likely click (if any)")
        axes[1].imshow(hierarchical_mask)

        for a in axes.flatten():
            a.axis("off")

        prediction_shape = np.array(hierarchical_mask.shape)
        most_likely_click = descriptions[k]["most_likely_click"]
        original_shape = descriptions[k]["original_shape"]

        if -1 not in most_likely_click:
            mlc_ii = most_likely_click * original_shape
            click_patch_ii = plt.Circle(mlc_ii[::-1], radius=2, color="green")
            axes[0].add_patch(click_patch_ii)

            mlc_hm = most_likely_click * prediction_shape
            click_patch_hm = plt.Circle(mlc_hm[::-1], radius=2, color="green")
            axes[1].add_patch(click_patch_hm)

        loop_present, r, c, h, w = descriptions[k]["aoi_bbox"]
        if loop_present != -1:
            try:
                r *= original_shape[0]
                c *= original_shape[1]
                h *= original_shape[0]
                w *= original_shape[1]
                C, R = int(c - w / 2), int(r - h / 2)
                W, H = int(w), int(h)
                loop_bbox_patch = plt.Rectangle(
                    (C, R), W, H, linewidth=1, edgecolor="green", facecolor="none"
                )
                axes[0].add_patch(loop_bbox_patch)
            except BaseException:
                pass

        comparison_path = prediction_img_path.replace(
            "hierarchical_mask_high_contrast_predicted", "comparison"
        )
        plt.savefig(comparison_path)
        plt.close()
        print("saving %s" % comparison_path)
    end = time.time()

    print(
        "%d predictions saved in %.4f seconds (%.4f per image)"
        % (len(input_images), end - _start, (end - _start) / len(input_images))
    )


def save_predictions(
    input_images,
    predictions,
    image_paths,
    ground_truths,
    notions,
    notion_indices,
    model_img_size,
    model_name="default",
    train=False,
    target=False,
    threshold=0.5,
    click_threshold=0.95,
):
    _start = time.time()
    for k, input_image in enumerate(input_images):
        hierarchical_mask = np.zeros(model_img_size, dtype=np.uint8)
        predicted_masks = np.zeros(model_img_size + (len(notions),), dtype=np.uint8)
        if "click" in notions:
            notions_in_order = notions[:-1][::-1] + [notions[-1]]
        else:
            notions_in_order = notions[::-1]
        for notion in notions_in_order:
            notion_value = notions.index(notion) + 1
            l = notion_indices[notion]
            if l != -1:
                mask = (predictions[l][k] > threshold)[:, :, 0]
                predicted_masks[:, :, l] = mask
            else:
                mask = (predictions[l][k] > click_threshold)[:, :, 0]
                predicted_masks[:, :, l] = mask
            if np.any(mask):
                hierarchical_mask[mask == 1] = notion_value
            hierarchical_mask[-1, -(1 + notions.index(notion))] = notion_value
        if target:
            label_mask = np.zeros(model_img_size, dtype=np.uint8)
            for notion in notions_in_order:
                notion_value = notions.index(notion) + 1
                l = notion_indices[notion]
                if l != -1:
                    mask = (ground_truths[l][k] > threshold)[:, :, 0]
                else:
                    mask = (predictions[l][k] > click_threshold)[:, :, 0]
                if np.any(mask):
                    label_mask[mask == 1] = notion_value
                label_mask[-1, -(1 + notions.index(notion))] = notion_value

        name = os.path.basename(image_paths[k])
        prefix = name[:-4]
        directory = os.path.dirname(image_paths[k])

        if train:
            prefix += "_train"

        template = "%s_%s_model_img_size_%dx%d" % (
            prefix,
            model_name.replace(".h5", ""),
            model_img_size[0],
            model_img_size[1],
        )

        prediction_img_path = os.path.join(
            directory, "%s_hierarchical_mask_high_contrast_predicted.png" % (template)
        )
        save_img(
            prediction_img_path, np.expand_dims(hierarchical_mask, axis=2), scale=True
        )

        predicted_masks_name = os.path.join(directory, "%s.npy" % template)
        np.save(predicted_masks_name, predicted_masks)

        if target:
            fig, axes = plt.subplots(1, 3)
        else:
            fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(16, 9)
        title = name
        fig.suptitle(title)
        axes[0].set_title(
            "input image with predicted click and loop bounding box (if any)"
        )
        axes[0].imshow(input_image)
        axes[1].set_title("raw segmentation result with most likely click (if any)")
        axes[1].imshow(hierarchical_mask)
        if target:
            axes[2].set_title("ground truth")
            axes[2].imshow(label_mask)

        for a in axes.flatten():
            a.axis("off")

        original_shape = np.array(input_image.shape[:2])
        prediction_shape = np.array(hierarchical_mask.shape)
        most_likely_click = np.array(get_most_likely_click(predictions, k=k))
        if -1 not in most_likely_click:
            mlc_ii = most_likely_click * original_shape
            click_patch_ii = plt.Circle(mlc_ii[::-1], radius=2, color="green")
            axes[0].add_patch(click_patch_ii)

            mlc_hm = most_likely_click * prediction_shape
            click_patch_hm = plt.Circle(mlc_hm[::-1], radius=2, color="green")
            axes[1].add_patch(click_patch_hm)

        loop_present, r, c, h, w = get_loop_bbox(predictions, k=k)
        if loop_present != -1:
            r *= original_shape[0]
            c *= original_shape[1]
            h *= original_shape[0]
            w *= original_shape[1]
            C, R = int(c - w / 2), int(r - h / 2)
            W, H = int(w), int(h)
            loop_bbox_patch = plt.Rectangle(
                (C, R), W, H, linewidth=1, edgecolor="green", facecolor="none"
            )
            axes[0].add_patch(loop_bbox_patch)

        comparison_path = prediction_img_path.replace(
            "hierarchical_mask_high_contrast_predicted", "comparison"
        )
        plt.savefig(comparison_path)
        plt.close()
        print("saving %s" % comparison_path)
    end = time.time()

    print(
        "%d predictions saved in %.4f seconds (%.4f per image)"
        % (len(input_images), end - _start, (end - _start) / len(input_images))
    )


def get_title_from_img_path(img_path):
    return os.path.basename(os.path.dirname(img_path))


def plot(
    sample,
    title="",
    k=0,
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "capillary",
        "ice",
        "foreground",
        "click",
    ],
):
    fig, axes = pylab.subplots(2, 5)
    fig.set_size_inches(24, 16)
    fig.suptitle(title)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")
    ax[0].imshow(sample[0][k])
    ax[0].set_title("input image")
    for l in range(len(sample[1])):
        ax[1 + l].imshow(sample[1][l][k][:, :, 0])
        ax[1 + l].set_title(notions[l])
    pylab.show()


def plot_augment(
    img_path,
    ntransformations=14,
    figsize=(24, 16),
    zoom_factor=0.5,
    shift_factor=0.5,
    shear_factor=45,
    rotate_probability=1,
    shear_probability=1,
    zoom_probability=1,
    shift_probability=1,
):
    fig, axes = pylab.subplots((2 * ntransformations) // 6 + 1, 6)
    fig.set_size_inches(*figsize)
    title = get_title_from_img_path(img_path)
    fig.suptitle(title)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")
    img, target = get_img_and_target(img_path)
    ax[0].imshow(img)
    ax[0].set_title("input image")
    ax[1].imshow(get_hierarchical_mask_from_target(target))
    ax[1].set_title("original_target")

    for t in range(1, ntransformations + 1):
        wimg = img[::]
        wtarget = target[::]
        if random.random() > 0.5:
            wimg, wtarget = get_flipped_img_and_target(wimg, wtarget)
        if random.random() > 0.5:
            wimg, wtarget = get_transposed_img_and_target(wimg, wtarget)

        wimg, wtarget = get_transformed_img_and_target(
            wimg,
            wtarget,
            shear_factor=shear_factor,
            zoom_factor=zoom_factor,
            shift_factor=shift_factor,
            rotate_probability=rotate_probability,
            shear_probability=shear_probability,
            zoom_probability=zoom_probability,
            shift_probability=shift_probability,
        )

        ax[2 * t].imshow(wimg)
        ax[2 * t].set_title("%d input" % (t + 1))
        ax[2 * t + 1].imshow(get_hierarchical_mask_from_target(wtarget))
        ax[2 * t + 1].set_title("%d target" % (t + 1))

    pylab.show()


def plot_batch(
    batch_size=16,
    transform=True,
    augment=True,
    swap_backgrounds=True,
    black_and_white=True,
    shuffle_at_0=True,
    flip=True,
    transpose=True,
    model_img_size=(256, 320),
    figsize=(24, 16),
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "capillary",
        "ice",
        "foreground",
        "hierarchy",
        "identity",
    ],
):
    paths, _ = get_training_and_validation_datasets(
        directory="images_and_labels", split=0.2
    )
    gen = get_generator(
        batch_size,
        model_img_size,
        paths,
        notions=notions,
        augment=augment,
        transform=transform,
        swap_backgrounds=swap_backgrounds,
        flip=flip,
        transpose=transpose,
        dynamic_batch_size=False,
        artificial_size_increase=False,
        shuffle_at_0=shuffle_at_0,
        black_and_white=black_and_white,
        verbose=True,
    )

    imgs, targets = gen[0]
    targets_as_multichannel_masks = np.zeros(
        imgs.shape[:3] + (len(gen.hierarchy_notions),), dtype="uint8"
    )
    for k in range(batch_size):
        for l in range(len(gen.hierarchy_notions)):
            targets_as_multichannel_masks[k, :, :, l] = targets[l][k, :, :, 0]
    fig, axes = pylab.subplots((2 * batch_size) // 6 + 1, 6)
    fig.set_size_inches(*figsize)
    title = "batch plot"
    fig.suptitle(title)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")
    for t in range(batch_size):
        ax[2 * t].imshow(imgs[t])
        ax[2 * t].set_title("%d input" % (t))
        ax[2 * t + 1].imshow(
            get_hierarchical_mask_from_target(targets_as_multichannel_masks[t])
        )
        ax[2 * t + 1].set_title("%d target" % (t))
    pylab.show()

def get_generator(paths,
                  batch_size=16,
                  model_img_size=(320, 256),
                  notions=[
                        "crystal",
                        "loop_inside",
                        "loop",
                        "stem",
                        "pin",
                        "capillary",
                        "ice",
                        "foreground",
                        "hierarchy",
                        "identity",
                    ],
                  augment=True,
                  transform=True,
                  swap_backgrounds=True,
                  black_and_white=True,
                  shuffle_at_0=True,
                  flip=True,
                  transpose=True,
                  dynamic_batch_size=False,
                  artificial_size_increase=False,
                  verbose=True,):
    
    gen = MultiTargetDataset(
        batch_size,
        model_img_size,
        paths,
        notions=notions,
        augment=augment,
        transform=transform,
        swap_backgrounds=swap_backgrounds,
        flip=flip,
        transpose=transpose,
        dynamic_batch_size=dynamic_batch_size,
        artificial_size_increase=artificial_size_increase,
        shuffle_at_0=shuffle_at_0,
        black_and_white=black_and_white,
        verbose=verbose,
    )
    return gen 

