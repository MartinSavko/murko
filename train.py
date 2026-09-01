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
    directories, seed=12345, split=0.2
):
    sample_families = get_sample_families(directories)
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

    train_paths = get_paths_for_families(train_families, sample_families)
    random.Random(seed).shuffle(train_paths)
    val_paths = get_paths_for_families(valid_families, sample_families)
    random.Random(seed).shuffle(val_paths)

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
        elif params[head["task"]]["metrics"] == "BIoUm":
            metrics[head_name] = [
                keras.metrics.BinaryIoUm(
                    target_class_ids=[1], threshold=0.5, name="BIoUm_1"
                ),
                keras.metrics.BinaryIoUm(
                    target_class_ids=[0], threshold=0.5, name="BIoUm_0"
                ),
                keras.metrics.BinaryIoUm(
                    target_class_ids=[0, 1], threshold=0.5, name="BIoUm_both"
                ),
            ]
        elif params[head["task"]]["metrics"] == "mean_absolute_error":
            metrics[head_name] = keras.metrics.MeanAbsoluteError(name="MAE")
        elif head["task"] == "hierarchy":
            metrics[head_name] = getattr(
                keras.metrics, params[head["task"]]["metrics"]
            )(head["channels"])

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
):
    if mixed_precision:
        print("setting mixed_precision")
        keras.mixed_precision.set_global_policy("mixed_float16")

    for gpu in tf.config.list_physical_devices("GPU"):
        print("setting memory_growth on", gpu)
        tf.config.experimental.set_memory_growth(gpu, True)

    tasks = [tc["name"] for tc in targets_config]
    distinguished_name = "%s_%s" % (network, name)
    model_name = os.path.join(base, "results", "%s.keras" % distinguished_name)
    history_name = os.path.join(base, "results", "%s.history" % distinguished_name)
    checkpoint_filepath = "%s_{batch:06d}_{loss:.4f}.keras" % distinguished_name
    tensorboard_dir = os.path.join(base, "results", "%s_logs" % distinguished_name)
    # segment_train_paths, segment_val_paths = get_training_and_validation_datasets()
    # print('training on %d samples, validating on %d samples' % ( len(train_paths), len(val_paths)))
    # data genrators
    train_paths, val_paths = get_training_and_validation_datasets(
        dataset, split=train_dev_split
    )
    # if include_plate_images:
    #     train_paths_plate, val_paths_plate = get_training_and_validation_datasets(
    #         dataset, split=0
    #     )
    #     # val_paths += val_paths_plate
    #     train_paths += train_paths_plate
    # if include_capillary_images:
    #     (
    #         train_paths_capillary,
    #         val_paths_capillary,
    #     ) = get_training_and_validation_datasets(
    #         dataset, split=0
    #     )
    #     # val_paths += val_paths_plate
    #     train_paths += train_paths_capillary
    #     val_paths += val_paths_capillary
    if train_dataset != []:
        train_paths +=  get_training_and_validation_datasets(
        train_dataset, split=0.
    )[0]

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
    pprint.pprint(f"tasks in train\n{tasks}")
    train_gen = JsonDataset(
        train_paths,
        targets_config,
        batch_size=batch_size,
        img_size=model_img_size,
        augment=augment,
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
    # val_gen = CrystalClickDataset(val_batch_size, val_model_img_size, val_paths, augment=False, scale_click=scale_click, click_radius=click_radius, dynamic_batch_size=False)
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
    tensorboard = keras.callbacks.TensorBoard(log_dir=tensorboard_dir, histogram_freq=1)
    
    callbacks = [checkpointer, nanterminator, lrreducer, tensorboard]
    network_parameters = networks[network]

    if os.path.isdir(model_name) or os.path.isfile(model_name):
        print("model exists, loading weights ...")
        # model = keras.models.load_model(model_name)
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
            finetune=finetune,
            finetune_model=model_name,
            limit_loss=limit_loss,
            weight_decay=weight_decay,
            activation=activation,
            **network_parameters,
        )
        if not finetune:
            try:
                model.load_weights(model_name)
            except:
                model = keras.models.load_model(
                    model_name,
                    # targets_config=targets_config,
                    custom_objects={
                        "WSConv2D": WSConv2D,
                        "WSSeparableConv2D": WSSeparableConv2D,
                    },
                )
        history_name = history_name.replace(".history", "_next_superepoch.history")
    else:
        print(model_name, "does not exist")
        # custom_objects = {"click_loss": click_loss, "ClickMetric": ClickMetric}
        # with keras.utils.custom_object_scope(custom_objects):
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
            **network_parameters,
        )

    print("model.summary()")
    # print(model.summary())

    pprint.pprint(f"targets_config\n{targets_config}")
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

    for candidate in targets_config:
        parser.add_argument(
            f'--{candidate["name"]}_{candidate["task"]}',
            default=1 if candidate in default_active else 0,
            type=int,
            help=f"learn {candidate}",
        )

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
        

    args = parser.parse_args()
    print("args", args)

    pixel_budget = int(args.pixel_budget * args.pixel_budget_modifier)
    model_img_size = get_img_size_as_scale_of_pixel_budget(args.resize_factor)
    val_model_img_size = eval(args.val_model_img_size)
    if args.batch_size == -1 and args.resize_factor != -1:
        model_img_size = get_img_size_as_scale_of_pixel_budget(args.resize_factor)
        if args.ratio == 1.0:
            model_img_size = (model_img_size[0], model_img_size[0])
        batch_size = get_dynamic_batch_size(model_img_size, pixel_budget)
        dynamic_batch_size = False
    elif args.batch_size == -1:
        dynamic_batch_size = True
        model_img_size = -1
        batch_size = args.batch_size
    else:
        model_img_size = eval(args.model_img_size)
        val_model_img_size = model_img_size
        batch_size = min(args.batch_size, get_dynamic_batch_size(model_img_size, pixel_budget))
        dynamic_batch_size = False
    print("model_img_size", model_img_size)
    print("val model_img_size", val_model_img_size)
    print("batch_size", batch_size)
    print("name: %s" % args.name)
    #sys.exit()
    # save the current version of the murko under a name corresponding to the
    # output model name
    for tool in ["murko", "train", "sample", "objects_of_interest", "regionprops", "dataset_loader"]:
        os.system("cp %s.py experiments/%s_%s_%s.py" % (tool, args.network, args.name, tool))

    f = open("experiments/%s_%s.args" % (args.network, args.name), "wb")
    pickle.dump(args, f)
    f.close()

    train(
        dataset=args.dataset,
        train_dataset=args.train_dataset,
        base=args.base,
        model_img_size=model_img_size,
        network=args.network,
        epochs=args.epochs,
        patience=args.patience,
        filter_size=args.filter_size,
        batch_size=batch_size,
        targets_config=targets_config,
        name=args.name,
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
    )


if __name__ == "__main__":
    main()
