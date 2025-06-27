#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

# Command line interface to set up the training process

import os
import sys
import pickle
import tensorflow as tf
from tensorflow import keras
import copy

from utils import plot_history

from murko import (
    params,
    networks,
    loss_weights_from_stats,
    get_uncompiled_tiramisu,
    get_num_segmentation_classes,
    WSConv2D,
    WSSeparableConv2D,
)

from dataset_loader import (
    get_dynamic_batch_size,
    get_img_size,
    get_training_and_validation_datasets,
    MultiTargetDataset,
)


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


def get_model(
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
        {"name": "crystal", "type": "binary_segmentation"},
        {"name": "loop_inside", "type": "binary_segmentation"},
        {"name": "loop", "type": "binary_segmentation"},
        {"name": "stem", "type": "binary_segmentation"},
        {"name": "pin", "type": "binary_segmentation"},
        {"name": "capillary", "type": "binary_segmentation"},
        {"name": "ice", "type": "binary_segmentation"},
        {"name": "foreground", "type": "binary_segmentation"},
        {"name": "click", "type": "click_segmentation"},
    ],
    name="model",
    normalization_type="GroupNormalization",
    limit_loss=True,
    weight_decay=1.0e-4,
):
    print("get_model heads", heads)
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
        print("head name and type", head["name"], head["type"])
        if params[head["type"]]["metrics"] == "BIoU":
            metrics[head["name"]] = [
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
        elif params[head["type"]]["metrics"] == "BIoUm":
            metrics[head["name"]] = [
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
        elif params[head["type"]]["metrics"] == "mean_absolute_error":
            metrics[head["name"]] = keras.metrics.MeanAbsoluteError(name="MAE")
        elif head["type"] == "categorical_segmentation":
            metrics[head["name"]] = getattr(
                keras.metrics, params[head["type"]]["metrics"]
            )(num_segmentation_classes + 1)

            # , sparse_y_true=True, sparse_y_pred=True)
            # losses[head["name"]] = keras.losses.BinaryFocalCrossentropy(name="hierarchy_loss", from_logits=True)
            # getattr(keras.losses, params[head["type"]]["loss"])(from_logits=True)
        else:
            metrics[head["name"]] = getattr(
                keras.metrics, params[head["type"]]["metrics"]
            )()

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
            lw = 1.0
        loss_weights[head["name"]] = lw

    print("loss weights", loss_weights)
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

    print("model.losses", len(model.losses), model.losses)
    print("model.metrics", len(model.metrics), model.metrics)
    return model


def train(
    base="/nfs/data2/Martin/Research/murko",
    epochs=25,
    patience=3,
    mixed_precision=False,
    name="start",
    source_weights=None,
    batch_size=16,
    model_img_size=(512, 512),
    network="fcdn103",
    convolution_type="SeparableConv2D",
    heads=[
        {"name": "crystal", "type": "binary_segmentation"},
        {"name": "loop_inside", "type": "binary_segmentation"},
        {"name": "loop", "type": "binary_segmentation"},
        {"name": "stem", "type": "binary_segmentation"},
        {"name": "pin", "type": "binary_segmentation"},
        {"name": "capillary", "type": "binary_segmentation"},
        {"name": "ice", "type": "binary_segmentation"},
        {"name": "foreground", "type": "binary_segmentation"},
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
    model_name = os.path.join(base, "results", "%s.keras" % distinguished_name)
    history_name = os.path.join(base, "results", "%s.history" % distinguished_name)
    checkpoint_filepath = "%s_{batch:06d}_{loss:.4f}.keras" % distinguished_name
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
    print("notions in train", notions)
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
        model = get_model(
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
            **network_parameters,
        )
        if not finetune:
            try:
                model.load_weights(model_name)
            except:
                model = keras.models.load_model(
                    model_name,
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
            **network_parameters,
        )

    print(model.summary())

    print(f"train_gen: {train_gen}")
    print(f"epochs: {epochs}")
    print(f"val_gen: {val_gen}")

    history = model.fit(
        train_gen,
        epochs=epochs,
        validation_data=val_gen,
        callbacks=callbacks,
        use_multiprocessing=True,
        workers=32,
        max_queue_size=128,
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

    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", default="tensorflow", type=str, help="backend")

    candidates = get_candidates()

    for candidate in candidates:
        parser.add_argument(
            "--%s" % candidate,
            default=1 if candidate in default_active else 0,
            type=int,
            help=f"learn {candidate} ({candidates[candidate]})",
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
    parser.add_argument("-e", "--epochs", default=1, type=int, help="numbers of epochs")
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
        "--val_model_img_size",
        default=(256, 320),
        type=tuple,
        help="validation img_model_size",
    )
    parser.add_argument(
        "--base",
        default="./",
        type=str,
        help="path to the directory where results will be saved",
    )

    args = parser.parse_args()
    print("args", args)

    heads = []
    for candidate in candidates:
        if bool(getattr(args, candidate)):
            heads.append({"name": candidate, "type": candidates[candidate]})

    print("heads", heads)

    pixel_budget = int(args.pixel_budget * args.pixel_budget_modifier)
    if args.batch_size == -1 and args.resize_factor != -1:
        model_img_size = get_img_size(args.resize_factor)
        if args.ratio == 1.0:
            model_img_size = (model_img_size[0], model_img_size[0])
        batch_size = get_dynamic_batch_size(model_img_size, pixel_budget)
        dynamic_batch_size = False
    elif args.batch_size == -1:
        dynamic_batch_size = True
        model_img_size = -1
        batch_size = args.batch_size
    else:
        dynamic_batch_size = False
    print("model_img_size", model_img_size)
    print("batch_size", batch_size)
    print("name: %s" % args.name)

    # save the current version of the murko under a name corresponding to the
    # output model name
    for tool in ["murko", "train", "dataset_loader"]:
        os.system("cp %s.py %s_%s_%s.py" % (tool, args.network, args.name, tool))

    f = open("%s_%s.args" % (args.network, args.name), "wb")
    pickle.dump(args, f)
    f.close()

    train(
        base=args.base,
        model_img_size=model_img_size,
        network=args.network,
        epochs=args.epochs,
        patience=args.patience,
        batch_size=batch_size,
        heads=heads,
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
        val_model_img_size=args.val_model_img_size,
    )


if __name__ == "__main__":
    main()
