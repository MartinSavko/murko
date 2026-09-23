#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

# Command line interface to set up the training process

import os
import sys
import subprocess
import re
import pickle
import random
import tensorflow as tf
from tensorflow import keras
import copy
import pprint
import numpy as np
from utils import plot_history

from murko import (
    params,
    networks,
    loss_weights_from_stats,
    get_uncompiled_tiramisu,
    WSConv2D,
    WSSeparableConv2D,
)

from dataset_loader import (
    get_dynamic_batch_size,
    get_img_size_as_scale_of_pixel_budget,
    JsonDataset,
)

from candidates import get_candidates
from objects_of_interest import get_objects_of_interest

# from Bing 2026-09-16
# query: masking loss function Keras
IGNORE_LABEL = -1
EPSILON = 1.e-7

def masked_categorical_crossentropy(y_true, y_pred):
    # Create mask for valid labels
    mask = tf.not_equal(y_true, IGNORE_LABEL)
    mask = tf.cast(mask, tf.float32)

    # Compute normal loss
    loss = tf.keras.losses.categorical_crossentropy(y_true, y_pred)

    # Apply mask
    loss = loss * mask
    return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + EPSILON)

def classification_categorical_accuracy(y_true, y_pred):
    _t = tf.reduce_mean(y_true, axis=(0, 1))
    _p = tf.reduce_mean(y_pred, axis=(0, 1))
    return tf.keras.metrics.categorical_accuracy(_t, _p)

# def categorical_accuracy(y_true, y_pred):
#     keras.metrics.CategoricalAccuracy()
#     acc = np.dot(sample_weight, np.equal(y_true, np.argmax(y_pred, axis=1))


# # query keras loss mask decorator
# from tensorflow.keras import backend as K
#
# def masked_loss(loss_fn):
#     """
#     Decorator to apply a mask to any loss function.
#     Assumes y_true has shape (..., features + 1) where the last feature is the mask.
#     """
#     def loss_with_mask(y_true, y_pred):
#         # Split mask from actual target
#         y_true_values = y_true[..., :-1]  # all but last column
#         mask = y_true[..., -1:]           # last column as mask (shape: batch, time, 1)
#
#         # Compute base loss
#         loss = loss_fn(y_true_values, y_pred)
#
#         # Ensure mask is same shape as loss
#         mask = tf.cast(mask, loss.dtype)
#         loss *= mask
#
#         # Avoid division by zero
#         return tf.reduce_sum(loss) / (tf.reduce_sum(mask) + K.epsilon())
#
#     return loss_with_mask
#
# # Example: wrap MSE with masking
# masked_mse = masked_loss(keras.losses.MeanSquaredError(reduction=tf.keras.losses.Reduction.NONE))


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

def check_directory(directory):
    if os.path.isdir(directory):
        pass
    else:
        os.makedirs(directory)

def save_pickled_file(filename, object_to_pickle, mode="wb"):
    check_directory(os.path.dirname(filename))
    f = open(filename, mode)
    pickle.dump(object_to_pickle, f)
    f.close()

def get_pickled_file(filename, mode="rb"):
    try:
        try:
            pickled_file = pickle.load(open(filename, mode))
        except:
            pickled_file = pickle.load(open(filename, mode), encoding="latin1")
    except IOError:
        pickled_file = None
    return pickled_file

def okay_for_validation(path):
    okay = True
    print("path", path)
    ooi = get_objects_of_interest(path)
    labels = ooi["labels"]
    if (
        "capillary" in labels
        or "foreground" in labels and len(labels) <= 2
    ):
        okay = False
    return okay

def prepare_train_and_validation_datasets(directory, split=0.2, valmax=100, force=False):

    train_name, valid_name = [os.path.join(directory, f"{n}_paths.pickle") for n in ["train", "valid"]]

    if force or not os.path.isfile(train_name) or not os.path.isfile(valid_name):


        t, v = get_training_and_validation_datasets([directory], split=split, valmax=valmax)

        not_wanted_for_validation = []
        replacements_for_validation = []
        for p in v:
            if not okay_for_validation(p):
                not_wanted_for_validation.append(p)
                np.random.shuffle(t)
                for _p in t:
                    if okay_for_validation(_p):
                        replacements_for_validation.append(_p)
                        break

        print('not_wanted_for_validation', not_wanted_for_validation)
        print('replacements_for_validation', replacements_for_validation)

        for nwfv in not_wanted_for_validation[::-1]:
            del v[v.index(nwfv)]
            t.append(nwfv)
        for rfv in replacements_for_validation:
            del t[t.index(rfv)]
            v.append(rfv)

        for p, n in zip([t, v], [train_name, valid_name]):
            save_pickled_file(n, p)

    else:
        t, v = [get_pickled_file(n) for n in [train_name, valid_name]]

    return t, v

def get_family(name):
    fname = os.path.realpath(name)
    # search_string = ".*/double_clicks_(.*)_double_click.*|.*/(.*)_manual_omega.*|.*/(.*)_color_zoom.*|.*/(.*)_auto_omega.*"
    search_string = ".*/double_clicks_(.*)_double_click.*|.*/(.*)_manual_omega.*|.*/(.*)_color_.*|.*/(.*)_auto_omega.*|.*/(.*)_click_.*"
    match = re.findall(search_string, fname)
    print("match", match)
    if match:
        for item in match[0]:
            if item != "":
                return item
    else:
        return os.path.basename(os.path.dirname(fname))


def get_individuals(directories):
    line = f'find {" ".join(directories)} -iname "*.json"'
    individuals = subprocess.getoutput(line).split("\n")

    return individuals

def get_sample_families(directories=["/nfs/data2/Martin/Research/murko/manually_segmented_images/json/spine/soleil_proxima2a"]):

    search_string = ".*/double_clicks_(.*)_double_click.*|.*/(.*)_manual_omega.*|.*/(.*)_color_.*|.*/(.*)_auto_omega.*|.*/(.*)_click_.*"
    individuals = get_individuals(directories)
    sample_families = {}
    for individual in individuals:
        matches = re.findall(search_string, individual)
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


def get_paths_for_families(families_subset_list, sample_families):
    paths = []
    for family in families_subset_list:
        for individual in sample_families[family]:
            paths.append(individual)
    return paths


def get_training_and_validation_datasets(
    directories, seed=12345, split=0.2, valmax=100, verbose=True,
):
    sample_families = get_sample_families(directories)
    sample_families_names = sorted(sample_families.keys())
    random.Random(seed).shuffle(sample_families_names)
    total = len(sample_families_names)

    train = int((1 - split) * total)
    train_families = sample_families_names[:train]
    valid_families = sample_families_names[train:]
    if verbose:
        print("total %d" % total)
        print("train", train)
        print("train_families: %d" % len(train_families))
        print("valid_families: %d" % len(valid_families))

    train_paths = get_paths_for_families(train_families, sample_families)
    val_paths = []
    while valid_families and len(val_paths) < valmax:
        family = valid_families.pop(0)
        for individual in sample_families[family]:
            val_paths.append(individual)

    train_paths += get_paths_for_families(valid_families, sample_families)

    return train_paths, val_paths


def get_model(
    nfilters=48,
    filter_size=3,
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
    targets_config=[
        {'name': 'crystal', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'loop_inside', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'loop', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'stem', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'pin', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'ice', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'foreground', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'area_of_interest', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'plastic', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'explorable', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'aether', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'crystal', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'loop_inside', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'loop', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'stem', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'pin', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'foreground', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'area_of_interest', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'plastic', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'explorable', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'aether', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'identity', 'task': 'encoder', 'dtype': 'float32', 'channels': 3, 'activation': 'sigmoid'},
        {'name': 'identity_bw', 'task': 'encoder', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'hierarchy_detailed', 'task': 'hierarchy', 'dtype': 'float32', 'channels': 7, 'concepts': ['background', 'foreground', 'pin', 'stem', 'loop', 'loop_inside', 'crystal'], 'activation': 'softmax'},
        {'name': 'hierarchy_crystal_aoi_support_pin', 'task': 'hierarchy', 'dtype': 'float32', 'channels': 6, 'concepts': ['background', 'foreground', 'pin', 'support', 'area_of_interest', 'crystal'], 'activation': 'softmax'},
        {'name': 'hierarchy_aoi', 'task': 'hierarchy', 'dtype': 'float32', 'channels': 3, 'concepts': ['background', 'foreground', 'area_of_interest'], 'activation': 'softmax'},
        {'name': 'hierarchy_crystal', 'task': 'hierarchy', 'dtype': 'float32', 'channels': 3, 'concepts': ['background', 'foreground', 'crystal'], 'activation': 'softmax'}
    ],
    name="model",
    normalization_type="GroupNormalization",
    limit_loss=True,
    weight_decay=1.0e-4,
    use_necks=False,
    neck_filters=16,
    neck_layers=4,
):
    print("get_model targets_config", targets_config)
    model = get_uncompiled_tiramisu(
        nfilters=nfilters,
        filter_size=filter_size,
        growth_rate=growth_rate,
        layers_scheme=layers_scheme,
        bottleneck=bottleneck,
        activation=activation,
        convolution_type=convolution_type,
        last_convolution=last_convolution,
        dropout_rate=dropout_rate,
        weight_standardization=weight_standardization,
        model_img_size=model_img_size,
        targets_config=targets_config,
        name=name,
        normalization_type=normalization_type,
        weight_decay=weight_decay,
        use_necks=use_necks,
        neck_filters=neck_filters,
        neck_layers=neck_layers,
    )
    if finetune and finetune_model is not None:
        print("loading weights to finetune")
        model.load_weights(finetune_model)
    else:
        print("not finetune")
    losses = {}
    metrics = {}

    for head in targets_config:
        head_name = f'{head["name"]}_{head["task"]}'
        losses[head_name] = params[head["task"]]["loss"]
        print("head name and type", head["name"], head["task"])
        if params[head["task"]]["metrics"] == "BIoU":
            metrics[head_name] = [
                keras.metrics.BinaryIoU(
                    target_class_ids=[1], threshold=0.5, name="BIoU_1"
                ),
                keras.metrics.BinaryIoU(
                    target_class_ids=[0], threshold=0.5, name="BIoU_0"
                ),
                keras.metrics.BinaryIoU(
                    target_class_ids=[0, 1], threshold=0.5, name="BIoU_both"
                ),
            ]
        elif params[head["task"]]["metrics"] == "mean_absolute_error":
            metrics[head_name] = keras.metrics.MeanAbsoluteError(name="MAE")
        elif head["task"] == "hierarchy":
            metrics[head_name] = getattr(
                keras.metrics, params[head["task"]]["metrics"]
            )(head["channels"])
        elif head["task"] == "classification":
            metrics[head_name] = getattr(
                keras.metrics, params[head["task"]]["metrics"]
            )()

            # , sparse_y_true=True, sparse_y_pred=True)
            # losses[head_name] = keras.losses.BinaryFocalCrossentropy(name="hierarchy_loss", from_logits=True)
            # getattr(keras.losses, params[head["task"]]["loss"])(from_logits=True)
        else:
            metrics[head_name] = getattr(
                keras.metrics, params[head["task"]]["metrics"]
            )()

    pprint.pprint(f"losses {len(losses)}\n{losses}")
    pprint.pprint(f"metrics {len(metrics)}\n{metrics}")

    loss_weights = {}
    for head in targets_config:
        head_name = f'{head["name"]}_{head["task"]}'
        if head["name"] in loss_weights_from_stats:
            lw = loss_weights_from_stats[head["name"]]
            if limit_loss:
                if lw > loss_weights_from_stats["crystal"]:
                    lw = loss_weights_from_stats["crystal"]
        else:
            lw = 1.0
        loss_weights[head_name] = lw

    #print("loss weights", loss_weights)
    lrs = learning_rate
    # lrs = keras.optimizers.schedules.ExponentialDecay(lrs, decay_steps=1e4, decay_rate=0.96, minimum_value=1e-7, staircase=True)
    optimizer = keras.optimizers.RMSprop(learning_rate=lrs)
    # optimizer = keras.optimizers.Adam(learning_rate=lrs)
    if finetune:
        for l in model.layers[: -len(heads)]:
            l.trainable = False

    model.compile(
        optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics
    )

    pprint.pprint(f"model.losses {len(model.losses)}\n{model.losses}\n")
    pprint.pprint(f"model.metrics {len(model.metrics)}\n{model.metrics}\n")
    return model


def train(
    dataset=["/nfs/data2/Martin/Research/murko/manually_segmented_images/json/spine/soleil_proxima2a"],
    train_dataset=[],
    base="./",
    experiments_dir="./experiments",
    results_dir="./results",
    epochs=25,
    patience=3,
    mixed_precision=False,
    name="start",
    source_weights=None,
    filter_size=3,
    batch_size=16,
    model_img_size=(512, 512),
    network="fcdn103",
    convolution_type="SeparableConv2D",
    targets_config=[
        {'name': 'crystal', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'loop_inside', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'loop', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'stem', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'pin', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'ice', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'foreground', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'area_of_interest', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'plastic', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'explorable', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'aether', 'task': 'binary_segment', 'dtype': 'int8', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'crystal', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'loop_inside', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'loop', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'stem', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'pin', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'foreground', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'area_of_interest', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'plastic', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'explorable', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'aether', 'task': 'distance_transform', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'identity', 'task': 'encoder', 'dtype': 'float32', 'channels': 3, 'activation': 'sigmoid'},
        {'name': 'identity_bw', 'task': 'encoder', 'dtype': 'float32', 'channels': 1, 'activation': 'sigmoid'},
        {'name': 'hierarchy_detailed', 'task': 'hierarchy', 'dtype': 'float32', 'channels': 7, 'concepts': ['background', 'foreground', 'pin', 'stem', 'loop', 'loop_inside', 'crystal'], 'activation': 'softmax'},
        {'name': 'hierarchy_crystal_aoi_support_pin', 'task': 'hierarchy', 'dtype': 'float32', 'channels': 6, 'concepts': ['background', 'foreground', 'pin', 'support', 'area_of_interest', 'crystal'], 'activation': 'softmax'},
        {'name': 'hierarchy_aoi', 'task': 'hierarchy', 'dtype': 'float32', 'channels': 3, 'concepts': ['background', 'foreground', 'area_of_interest'], 'activation': 'softmax'},
        {'name': 'hierarchy_crystal', 'task': 'hierarchy', 'dtype': 'float32', 'channels': 3, 'concepts': ['background', 'foreground', 'crystal'], 'activation': 'softmax'}
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
    val_model_img_size=(256, 256),
    max_queue_size=128,
    workers=32,
    use_multiprocessing=True,
    do_transform=False,
    save_paths_and_config=True,
    use_necks=False,
    neck_filters=16,
    neck_layers=4,
):
    if mixed_precision:
        print("setting mixed_precision")
        keras.mixed_precision.set_global_policy("mixed_float16")

    for gpu in tf.config.list_physical_devices("GPU"):
        print("setting memory_growth on", gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    tasks = [tc["name"] for tc in targets_config]
    distinguished_name = "%s_%s" % (network, name)
    model_name = os.path.join(results_dir, "model.keras")
    history_name = os.path.join(results_dir, "history.pickle")
    checkpoint_filepath = os.path.join(results_dir, "model_{batch:06d}_{loss:.4f}.keras")
    tensorboard_dir = os.path.join(results_dir, "logs")

    network_parameters = networks[network]

    model = get_model(
        convolution_type=convolution_type,
        filter_size=filter_size,
        model_img_size=(None, None),
        targets_config=targets_config,
        last_convolution=last_convolution,
        name=network,
        learning_rate=learning_rate,
        dropout_rate=dropout_rate,
        weight_standardization=weight_standardization,
        normalization_type=normalization_type,
        limit_loss=limit_loss,
        weight_decay=weight_decay,
        activation=activation,
        use_necks=use_necks,
        neck_filters=neck_filters,
        neck_layers=neck_layers,
        **network_parameters,
    )

    if os.path.isdir(model_name) or os.path.isfile(model_name):
        print("model exists, loading weights ...")
        model.load_weights(model_name)

    print("model.summary()")
    # print(model.summary())

    train_paths, val_paths = [], []
    for d in dataset:
        t, v = prepare_train_and_validation_datasets(
            d, split=train_dev_split
        )
        train_paths += t
        val_paths += v

    if train_dataset != []:
        train_paths += get_training_and_validation_datasets(
            train_dataset, split=0.
        )[0]


    if save_paths_and_config:
        for p, n in zip([train_paths, val_paths, targets_config], ["train_paths", "val_paths", "targets_config"]):
            f = open(f"{experiments_dir}/{distinguished_name}_{n}.pickle", "wb")
            pickle.dump(p, f)
            f.close()

    full_size = len(train_paths)
    if train_images != -1:
        train_paths = train_paths[:train_images]
        factor = full_size // len(train_paths)
        train_paths = train_paths * (factor + 1)
        random.Random(seed).shuffle(train_paths)
        train_paths = train_paths[:full_size]

    print("\ntotal number of samples %d" % len(train_paths + val_paths))
    print(
        "training on %d samples, validating on %d samples\n"
        % (len(train_paths), len(val_paths))
    )

    # data genrators
    pprint.pprint(f"tasks in train\n{tasks}")
    train_gen = JsonDataset(
        train_paths,
        targets_config,
        batch_size=batch_size,
        img_size=model_img_size,
        augment=augment,
        do_transform=do_transform,
        dynamic_batch_size=dynamic_batch_size,
        pixel_budget=pixel_budget,
        artificial_size_increase=artificial_size_increase,
        shuffle_at_0=True,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
    )
    if val_model_img_size is None:
        val_model_img_size = get_img_size_as_scale_of_pixel_budget(validation_scale)
    val_batch_size = get_dynamic_batch_size(val_model_img_size)
    print("validation model_img_size will be", val_model_img_size)

    val_gen = JsonDataset(
        val_paths,
        targets_config,
        batch_size=val_batch_size,
        img_size=val_model_img_size,
        augment=False,
        pixel_budget=pixel_budget,
        max_queue_size=max_queue_size,
        workers=workers,
        use_multiprocessing=use_multiprocessing,
    )

    # callbacks
    checkpointer = keras.callbacks.ModelCheckpoint(
        model_name, verbose=1, monitor="val_loss", save_best_only=True, mode="min"
    )
    # checkpointer2 = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, monitor='loss', save_freq=2000, save_best_only=False, mode='min')

    nanterminator = keras.callbacks.TerminateOnNaN()

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
    tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)

    callbacks = [checkpointer, nanterminator, lrreducer, tensorboard]


    print(f"train_gen: {train_gen}")
    print(f"epochs: {epochs}")
    print(f"val_gen: {val_gen}")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
    )

    f = open(history_name, "wb")
    pickle.dump(history.history, f)
    f.close()

    plot_history(history_name, history.history)


def main():

    default_active = [
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "area_of_interest",
        "support",
        "explorable",
        "drop",
        # "precipitate"
        "hierarchy",
        "identity",
        "identity_bw",
        "foreground",
        "aether",
    ]

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        # https://stackoverflow.com/questions/36166225/using-the-same-option-multiple-times-in-pythons-argparse
        "-d",
        "--dataset",
        default=["/nfs/data2/Martin/Research/murko/manually_segmented_images/json/spine/soleil_proxima2a"],
        nargs="+",
        # action="append",
        # type=str,
        help="dataset",
    )
    parser.add_argument(
        # https://stackoverflow.com/questions/36166225/using-the-same-option-multiple-times-in-pythons-argparse
        "--train_dataset",
        default=[],
        nargs="*",
        # action="append",
        # type=str,
        help="additional datasets for training",
    )

    parser.add_argument("--backend", default="tensorflow", type=str, help="backend")

    targets_config, task_concepts = get_candidates()
    pprint.pprint(f"targets_config\n{targets_config}")
    pprint.pprint(f"task_concepts\n{task_concepts}")
    # pprint.pprint(target_config)
    # print("task_concepts", task_concepts)

    # for candidate in targets_config:
    #     parser.add_argument(
    #         f'--{candidate["name"]}_{candidate["task"]}',
    #         default=1 if candidate in default_active else 0,
    #         type=int,
    #         help=f"learn {candidate}",
    #     )

    parser.add_argument(
        "-r",
        "--resize_factor",
        default=-1,
        type=float,
        help="resize factor to use, original size ~1024x1360",
    )
    parser.add_argument("-R", "--ratio", default=1.0, type=float, help="H/W ratio")
    parser.add_argument(
        "-n", "--network", default="fcdn103", help="network architecture"
    )
    parser.add_argument(
        "-t", "--train_images", default=-1, type=int, help="number of training images"
    )
    parser.add_argument(
        "-v",
        "--valid_images",
        default=10000,
        type=int,
        help="number of validation images",
    )
    parser.add_argument(
        "-s", "--scale_click", default=0, type=int, help="scale the click with the zoom"
    )
    parser.add_argument(
        "-m", "--mixed_precision", default=1, type=int, help="use mixed_precision"
    )

    parser.add_argument(
        "--filter_size",
        default=3,
        type=int,
        help="filter_size",
    )
    parser.add_argument(
        "-b",
        "--batch_size",
        default=-1,
        type=int,
        help="batch size to use (-1 by default -- will try to do something intelligent about choosing the right size, either maximum that fix into memory or a dynamic one, again based on an model image size and available memory).",
    )
    parser.add_argument(
        "-c", "--click_radius", default=0.320, type=float, help="click radius in mm"
    )
    parser.add_argument(
        "-a", "--augment", default=1, type=int, help="augment during training"
    )
    parser.add_argument("-e", "--epochs", default=3, type=int, help="numbers of epochs")
    parser.add_argument(
        "-l", "--learning_rate", default=0.001, type=float, help="initial learning rate"
    )
    parser.add_argument(
        "-p", "--pixel_budget", default=768 * 992, type=int, help="pixel budget"
    )
    parser.add_argument(
        "-I",
        "--pixel_budget_modifier",
        default=1.0,
        type=float,
        help="pixel budget modifier",
    )
    parser.add_argument(
        "-N",
        "--normalization_type",
        default="GroupNormalization",
        type=str,
        help="normalization type to use",
    )
    parser.add_argument(
        "-A", "--name", default="test", type=str, help="name of the model"
    )
    parser.add_argument("-f", "--finetune", default=0, type=int, help="finetune")
    parser.add_argument(
        "-P", "--patience", default=2, type=int, help="patience for lrreducer"
    )

    parser.add_argument(
        "-i",
        "--artificial_size_increase",
        default=1,
        type=int,
        help="artificial size increase, integer",
    )
    parser.add_argument(
        "-H", "--include_plate_images", default=0, type=int, help="include plate images"
    )
    parser.add_argument(
        "-C",
        "--include_capillary_images",
        default=0,
        type=int,
        help="include capillary images",
    )
    parser.add_argument(
        "-T",
        "--convolution_type",
        default="SeparableConv2D",
        type=str,
        help="convolution_type",
    )
    parser.add_argument(
        "-W",
        "--weight_standardization",
        default=1,
        type=int,
        help="whether to apply weight standardization",
    )
    parser.add_argument(
        "-D", "--dropout_rate", default=0.2, type=float, help="dropout_rate"
    )
    parser.add_argument("-L", "--limit_loss", default=1, type=int, help="limit loss")
    parser.add_argument(
        "-w", "--weight_decay", default=1e-4, type=float, help="weight_decay"
    )
    parser.add_argument(
        "-V", "--activation", default="relu", type=str, help="activation"
    )
    parser.add_argument(
        "--train_dev_split", default=0.2, type=float, help="train dev split"
    )

    parser.add_argument(
        "--model_img_size",
        default="(256, 256)",
        type=str,
        help="train model_img_size",
    )

    parser.add_argument(
        "--val_model_img_size",
        default="(256, 256)",
        type=str,
        help="validation model_img_size",
    )
    parser.add_argument(
        "--base",
        default="./",
        type=str,
        help="path to the directory where results will be saved",
    )
    
    parser.add_argument(
        "--workers",
        default=32,
        type=int,
        help="workers",
    )
    parser.add_argument(
        "--max_queue_size",
        default=128,
        type=int,
        help="max_queue_size",
    )
    parser.add_argument(
        "--not_multiprocessing",
        action="store_true",
        help="do not use multiprocessing",
    )
        
    parser.add_argument(
        "--dont_transform",
        action="store_true",
        help="do not do random transform as part of data augmentation.",
    )

    parser.add_argument(
        "--use_necks",
        action="store_true",
        help="use necks",
    )

    parser.add_argument(
        "--neck_filters",
        default=16,
        type=int,
        help="neck filters",
    )

    parser.add_argument(
        "--neck_layers",
        default=4,
        type=int,
        help="neck layers",
    )

    args = parser.parse_args()
    print("args", args)

    pixel_budget = int(args.pixel_budget * args.pixel_budget_modifier)
    # model_img_size = eval(args.model_img_size) #get_img_size_as_scale_of_pixel_budget(args.resize_factor)
    # # val_model_img_size = eval(args.val_model_img_size)
    # val_model_img_size = model_img_size
    # if args.batch_size == -1 and args.resize_factor != -1:
    #     # model_img_size = get_img_size_as_scale_of_pixel_budget(args.resize_factor)
    #     # if args.ratio == 1.0:
    #     #     model_img_size = (model_img_size[0], model_img_size[0])
    #     batch_size = get_dynamic_batch_size(model_img_size, pixel_budget)
    #     dynamic_batch_size = False
    # elif args.batch_size == -1:
    #     dynamic_batch_size = True
    #     # model_img_size = -1
    #     # batch_size = args.batch_size
    # else:
    model_img_size = eval(args.model_img_size)
    val_model_img_size = model_img_size
    batch_size = max(3, get_dynamic_batch_size(model_img_size, pixel_budget))
    dynamic_batch_size = False
    print("model_img_size", model_img_size)
    print("val model_img_size", val_model_img_size)
    print("batch_size", batch_size)
    name = args.name + f"_b_{batch_size}"
    print("name: %s" % name)

    # sys.exit()
    # save the current version of the murko under a name corresponding to the
    # output model name
    experiments_dir = os.path.join(args.base, "experiments", f"{args.network}_{name}")
    results_dir = os.path.join(args.base, "experiments", f"{args.network}_{name}")
    for d in [experiments_dir, results_dir]:
        if not os.path.isdir(d):
            os.makedirs(d)

    for tool in ["murko", "train", "sample", "objects_of_interest", "regionprops", "dataset_loader"]:
        os.system(f"cp {tool}.py {experiments_dir}/{tool}.py")

    f = open(f"{experiments_dir}/args.pickle", "wb")
    pickle.dump(args, f)
    f.close()

    train(
        dataset=args.dataset,
        train_dataset=args.train_dataset,
        base=args.base,
        experiments_dir=experiments_dir,
        results_dir=results_dir,
        model_img_size=model_img_size,
        network=args.network,
        epochs=args.epochs,
        patience=args.patience,
        filter_size=args.filter_size,
        batch_size=batch_size,
        targets_config=targets_config,
        name=name,
        mixed_precision=args.mixed_precision,
        augment=bool(args.augment),
        train_images=args.train_images,
        valid_images=args.valid_images,
        scale_click=bool(args.scale_click),
        click_radius=args.click_radius,
        learning_rate=args.learning_rate,
        pixel_budget=pixel_budget,
        normalization_type=args.normalization_type,
        dynamic_batch_size=dynamic_batch_size,
        finetune=bool(args.finetune),
        artificial_size_increase=args.artificial_size_increase,
        include_plate_images=bool(args.include_plate_images),
        include_capillary_images=bool(args.include_capillary_images),
        convolution_type=args.convolution_type,
        dropout_rate=args.dropout_rate,
        weight_standardization=bool(args.weight_standardization),
        limit_loss=bool(args.limit_loss),
        weight_decay=args.weight_decay,
        activation=args.activation,
        train_dev_split=args.train_dev_split,
        val_model_img_size=val_model_img_size,
        max_queue_size=args.max_queue_size,
        workers=args.workers,
        use_multiprocessing=not args.not_multiprocessing,
        do_transform=not args.dont_transform,
        use_necks=args.use_necks,
        neck_filters=args.neck_filters,
        neck_layers=args.neck_layers,
    )


if __name__ == "__main__":
    main()
