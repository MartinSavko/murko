#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

# Command line interface to set up the training process

import os
import sys
import pickle
from murko import segment_multihead
from dataset_loader import get_dynamic_batch_size, get_img_size


def train():

    default_candidates = ["crystal", "loop_inside", "loop", "stem", "pin", "foreground"]

    candidates = dict(
        [
            ("crystal", "segmentation"),
            ("loop_inside", "segmentation"),
            ("loop", "segmentation"),
            ("stem", "segmentation"),
            ("pin", "segmentation"),
            ("ice", "segmentation"),
            ("capillary", "segmentation"),
            ("foreground", "segmentation"),
            ("hierarchy", "categorical_segmentation"),
            ("identity", "regression"),
            ("click", "regression"),
            ("crystal_bbox", "regression"),
            ("loop_inside_bbox", "regression"),
            ("loop_bbox", "regression"),
            ("stem_bbox", "regression"),
            ("pin_bbox", "regression"),
        ]
    )

    import argparse

    parser = argparse.ArgumentParser()
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
        "-A",
        "--name",
        default="test",
        type=str,
        help="name of the model",
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
        "-H",
        "--include_plate_images",
        default=0,
        type=int,
        help="include plate images",
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

    for candidate in candidates:
        parser.add_argument(
            "--%s" % candidate,
            default=1 if candidate in default_candidates else 0,
            type=int,
            help="learn %s" % candidate,
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
    for tool in ["murko", "train.py", "dataset_loader"]:
        os.system("cp %s.py %s_%s_%s.py" % (tool, args.network, args.name, tool))

    f = open("%s_%s.args" % (args.network, args.name), "wb")
    pickle.dump(args, f)
    f.close()

    segment_multihead(
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
    train()
