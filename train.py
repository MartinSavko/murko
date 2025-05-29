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
        "drop",
        #"precipitate"
        "hierarchy",
        "identity",
        "identity_bw",
        "foreground",
    ]

    binary_segmentations = [
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "ice",
        "capillary",
        "foreground",
        "aether",
        "explorable",
        "support",
        "area_of_interest",
        "drop",
        #"precipitate",
        "diffracting_area",  # from raster scans
    ]

    distance_transforms = copy.copy(binary_segmentations)
    for notion in ["ice", "diffracting_area"]:
        del distance_transforms[distance_transforms.index(notion)]

    bounding_boxes = copy.copy(binary_segmentations)
    for notion in ["ice", "diffracting_area", "aether"]: # "support", "foreground"
        del bounding_boxes[bounding_boxes.index(notion)]

    centerness = copy.copy(binary_segmentations)
    for notion in ["ice", "diffracting_area", "aether"]:
        del centerness[centerness.index(notion)

    '''
    How to represent and learn points ?
    I see two options: 
      a) lear offset(s) from point for each learning location i.e. have two location maps, one for horizontal offset and one for vertical offset. 
            1.) have separate head for each of the point categories. i.e. "loop_inner_point" would be learned separately from "crystal_inner_point". ( But what if we have more then one crystal in the image?)
            2.) each category of points has one learing output i.e. "inner_centers"
           each learning location than learns offset to the nearest inner point.
        b) learn 2d representation of point locations, 
            1.) a supperposition of gaussians for "inner_centers", "centroids", "bbox_centers". 
                What about "extreme_points", "eigen_points" and "global_keypoints"? *) Each subcategory e.g. "extreme_point_topmost" or "extreme" or "start_likely" has a separate output.
                **) "extreme_points" has one output (four peak gaussian mixture)
                    "start_likely" has one output (one gaussian)
            2.) offset learners they learn x, y distance to the nearest keypoint
    
    What should offset learners do if there is no point of interest present?
      a) do nothing ?
      b) learn to report there is nothing by returning something specific e.g. -1 ?
    
    Learning bbox_centers? (centerness)
             inner_centers?
            
    '''
    
    inner_centers = copy.copy(binary_segmentations)
    for notion in ["ice", "diffracting_area", "aether"]: # "support", "foreground"
        del inner_centers[inner_centers.index(notion)]

    extreme_points = copy.copy(binary_segmentations)
    for notion in ["ice", "diffracting_area", "aether"]:
        del extreme_points[extreme_points.index(notion)]

    eigen_points = copy.copy(binary_segmentations)
    for notion in ["ice", "diffracting_area", "aether"]:
        del eigen_points[eigen_points.index(notion)]

    encoded_shapes = copy.copy(binary_segmentations)
    for notion in ["ice", "diffracting_area", "aether", "support", "foreground"]:
        del encoded_shapes[encoded_shapes.index(notion)]

    categorical = ["hierarchy"]

    encoders = ["identity", "identity_bw"]

    points = [
        "most_likely_click",
        "extreme",
        "end_likely",
        "start_likely",
        "start_possible",
        "origin",
    ]

    candidates = []
    for item in binary_segmentations:
        candidates.append((f"{item}_binary_segmentation", "binary_segmentation")) # clear
    for item in distance_transforms:
        candidates.append((f"{item}_distance_transform", "distance_transform")) # clear, what about sqrt(dt), 1-dt, sqrt(1-dt) ?
    for item in distance_transforms:
        candidates.append((f"{item}_inverse_distance_transform", "inverse_distance_transform")) # clear
    for item in distance_transforms:
        candidates.append((f"{item}_sqrt_distance_transform", "inverse_distance_transform")) #clear
    for item in distance_transforms:
        candidates.append((f"{item}_inverse_sqrt_distance_transform", "inverse_distance_transform")) #clear
    for item in bounding_boxes:
        candidates.append((f"{item}_bbox", "bounding_box")) # learn four layer output (ltrb) separately, learn associated centerness separately
    for item in centerness:
        candidates.append((f"{item}_centerness", "centerness")) # clear binary cross entropy, or focal loss, modified centerness d = (1 - centerness**2)**2
    for item in inner_centers:
        candidates.append((f"{item}_inner_center", "inner_center")) # centerness, offsets, heatmap, distance, (1 - distance), sqrt(1-distance), 
        # for point p
        # xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        # d = np.sqrt((p[0]-yv)**2 + (p[1]-xv)**2)
        # d = d / d.max()
        # d = (1 - d)**2
        # if p not present d = -1
    for item in extreme_points:
        candidates.append((f"{item}_extreme_points", "extreme_points")) 
        # heatmap for every class of objects and every type of point 
        # + offset to the center_of_mass (x, y, 2 layers)
        # + size of the object (width and height, 2 layers)
        # + major_axis, minor_axis (2 layers)
        # + offset to the center of ellipse (x, y, 2 layers)
        # + orientation (8 layers) according to Mousavian, or a single number?
        # + area of the object (1 layer)
    for item in eigen_points:
        candidates.append((f"{item}_eigen_points", "eigen_points")) # heatmap for every class of objects and every type of point + offset to the center_of_mass (2 layers) + size of the object (width and height) + area of the object (1 layer)
    for item in encoded_shapes:
        candidates.append((f"{item}_shape", "encoded_shape")) # C (e.g. C=20) layer output
    for item in categorical:
        candidates.append((item, "categorical_segmentation")) # clear
    for item in encoders:
        candidates.append((item, "encoder")) # clear
    for item in points:
        candidates.append((item, "point"))

    candidates = dict(candidates)

    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument("--backend", default="tensorflow", type=str, help="backend")

    for candidate in candidates:
        parser.add_argument(
            "--%s" % candidate,
            default=1 if candidate in default_active else 0,
            type=int,
            help="learn %s" % candidate,
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
