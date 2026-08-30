#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import re
import time
import numpy as np

import tensorflow as tf
from tensorflow import keras

from murko import (
    WSConv2D,
    WSSeparableConv2D,
)

from dataset_loader import plot_image_and_targets
from candidates import get_candidates
from show_annotations import test_dataset_loader, plot_random_sample_and_targets


def test(model, dl, targets_config, model_designation, batch_size=16, verbose=False):
    dl.target = False
    
    start = time.time()
    ps = model.predict(dl, batch_size=batch_size)
    end = time.time()
    print(f"prediction of {len(dl)} samples took {end-start:.3f} seconds")
        
    start = time.time()
    for k in range(len(dl)):
        img = dl.samples[k].get_image()
        path = dl.samples[k].json_path
        if verbose: print(f'{k+1} of {len(dl)} {path}')
        targets = [ps[l][k] for l in range(len(ps))]
        plot_image_and_targets(img, targets, targets_config, path=dl.
    samples[k].json_path, save=True, threshold=0.5, model_designation=model_designation, close=True)
    end = time.time()
    print(f"generation of {len(dl)} reports took {end-start:.3f} seconds")
        
def main():
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    
    parser.add_argument(
        "-m",
        "--model_name",
        type=str,
        default="results/fcdn103_p2_fixed_64x64_fs_7_b_128_no_gamma_no_blur_no_noise_only_shifts_with_arthur_validated.keras",
        help="model",
    )
    
    parser.add_argument(
        "--model_img_size",
        default=None,
        type=str,
        help="train model_img_size",
    )
        
    parser.add_argument(
        "-a",
        "--augment",
        action="store_true",
        help="augment",
    )
    
    parser.add_argument(
        "-v",
        "--verbose",
        action="store_true",
        help="verbose",
    )
    
    parser.add_argument(
        "-b",
        "--batch_size",
        type=int,
        default=16,
        help="batch size",
    )
    
    parser.add_argument(
        "-d",
        "--model_designation",
        default=None,
        type=str,
        help="model designation",
    )
    
    parser.add_argument(
        "-w",
        "--warmup",
        action="store_true",
        help="warmup",
    )
    
    args = parser.parse_args()
    print(args)
    
    for gpu in tf.config.list_physical_devices("GPU"):
        print("setting memory_growth on", gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    
    if args.model_designation is None:
        model_designation = "_" + os.path.basename(args.model_name).replace(".keras", "")
    else:
        model_designation = args.model_designation
        
    if args.model_img_size is not None:
        model_img_size = eval(args.model_img_size)
    else:
        model_img_size = tuple(map(int, re.findall(".*_(\d+x\d+)_.*", args.model_name)[0].split("x")))
        
    print(f"model_img_size is {model_img_size}")
    
    dl = test_dataset_loader(
        img_size = model_img_size,
        augment=args.augment,
        verbose=args.verbose,
        batch_size=1,
    )
    
    model = keras.models.load_model(
        args.model_name,
        custom_objects={
            "WSConv2D": WSConv2D,
            "WSSeparableConv2D": WSSeparableConv2D,
        },
    )
    
    if args.warmup:
        start = time.time()
        warmup = model.predict((np.random.random((args.batch_size,) + model_img_size + (3,))), batch_size=args.batch_size)
        end = time.time()
        print(f"loading and warmup of {args.model_name} model took {end-start:.3f} seconds")
    
    targets_config, concepts = get_candidates()
    
    test(model, dl, targets_config, model_designation, batch_size=args.batch_size, verbose=args.verbose)
    
if __name__ == "__main__":
    main()
