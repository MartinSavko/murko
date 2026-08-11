#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import copy
import copy
import math
import random
import numpy as np

from keras.utils import Sequence
from sample import Sample
from candidates import get_candidates

# from config import targets_config

def get_dynamic_batch_size(img_size, pixel_budget=768 * 992):
    return max(int(pixel_budget / np.prod(img_size)), 1)


def get_batch(i, img_paths, batch_size):
    half, r = divmod(batch_size, 2)
    indices = np.arange(i - half, i + half + r)
    return [img_paths[divmod(item, len(img_paths))[1]] for item in indices]


def get_dynamic_batch_size(img_size, pixel_budget=768 * 992):
    return max(int(pixel_budget / np.prod(img_size)), 1)


def get_img_size_as_scale_of_pixel_budget(
    scale, pixel_budget=768 * 992, ratio=0.75, modulo=32
):
    n = math.floor(math.sqrt(pixel_budget / ratio))
    new_n = n * scale
    img_size = np.array((new_n * ratio, new_n)).astype(int)
    img_size -= np.mod(img_size, modulo)
    return tuple(img_size)

from show_annotations import timeit

@timeit
def load_samples_from_annotations(annotations):
    samples = [Sample(item) for item in annotations]
    return samples

class JsonDataset(Sequence):
    
    def __init__(
        self,
        annotations,
        targets_config,
        task_concepts,
        batch_size=1,
        dynamic_batch_size=False,
        number_batch_size_scales=32,
        img_size=(256, 320),
        possible_ratios=[0.75, 1.0],
        augment=False,
        swap_backgrounds=True,
        min_scale=0.15,
        max_scale=1.0,
        pixel_budget=768 * 992,
        artificial_size_increase=1,
        shuffle_at_0=False,
        target=True,
        verbose=False,
        workers=10,
        use_multiprocessing=True,
        max_queue_size=10,
    ):

        self.annotations = annotations
        self.targets_config = targets_config
        self.task_concepts = task_concepts

        self.batch_size = batch_size
        self.dynamic_batch_size = dynamic_batch_size
        if self.dynamic_batch_size:
            self.batch_size = 1
        self.possible_scales = np.linspace(
            min_scale, max_scale, number_batch_size_scales
        )
        self.img_size = img_size
        self.possible_ratios = possible_ratios

        # augmentation parameters
        self.augment = augment
        self.swap_backgrounds = swap_backgrounds
        self.pixel_budget = pixel_budget
        
        self.artificial_size_increase = artificial_size_increase
        if artificial_size_increase > 1:
            annotations = annotations * int(artificial_size_increase)

        self.samples = load_samples_from_annotations(annotations)
        self.nsamples = len(self.samples)

        if self.swap_backgrounds:
            # self.backgrounds = [
            #     sample
            #     for sample in self.samples
            #     if "background" in sample.image_path.lower()
            # ]
            self.backgrounds = [
                sample
                for sample in self.samples
                if "foreground" not in sample.labels
            ]

        self.shuffle_at_0 = shuffle_at_0
        self.target = target

        self.verbose = verbose

        super().__init__(
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )

    def __len__(self):
        return self.nsamples

    def get_empty_sample(self, img_size):
        y = []
        for target in self.targets_config:
            output = np.zeros(img_size + (target["channels"],), dtype=target["dtype"])
            y.append(output)
        return y

    def get_empty_batch(self, batch_size, img_size):
        y = []
        for target in self.targets_config:
            output = np.zeros(
                (batch_size,) + img_size + (target["channels"],),
                dtype=target["dtype"],
            )
            y.append(output)
        return y

    def get_img_size_and_batch(self, idx):
        if self.dynamic_batch_size:
            img_size = get_img_size_as_scale_of_pixel_budget(
                random.choice(self.possible_scales),
                pixel_budget=self.pixel_budget,
                ratio=random.choice(self.possible_ratios),
            )
            batch_size = get_dynamic_batch_size(
                img_size, pixel_budget=self.pixel_budget
            )
            batch = get_batch(idx, self.samples, batch_size)
        else:
            img_size = self.img_size[:]
            batch_size = self.batch_size
            start_index = idx * self.batch_size
            end_index = start_index + batch_size
            batch = self.samples[start_index: end_index]

        return img_size, batch

    def __getitem__(self, idx):
        if idx == 0 and self.shuffle_at_0:
            random.shuffle(self.samples)

        img_size, batch = self.get_img_size_and_batch(idx)
        
        batch_size = len(batch)

        x = np.zeros((batch_size,) + img_size + (3,), dtype="float32")
        if self.target:
            y = self.get_empty_batch(batch_size, img_size)

        for j, sample in enumerate(batch):
            img, targets = self.get_image_and_targets(sample, img_size)
            x[j] = img
            if self.target:
                for k, target in enumerate(targets):
                    y[k][j] = target[k]

        if self.target and len(y) == 1:
            y = y[0]

        return x, y if self.target else x

    def get_image_and_targets(self, sample, img_size, new_background=None):

        if self.augment and self.swap_backgrounds:
            new_background = random.choice(self.backgrounds)["image"]
        
        img, points = sample.get_image_and_points(
            img_size=img_size, augment=self.augment, new_background=new_background
        )

        if not self.target: return img

        targets = get_targets(sample, img, points, self.task_concepts, self.targets_config, self.augment, new_background=new_background)

        return img, targets

def pre_computable(concept):
    return "binary_segment" in concept or "distance_transform" in concept or "centerness" in concept

@timeit
def pre_compute(sample, img, points, task_concepts):
    pre_computed = {}
    for concept in task_concepts:
        if pre_computable(concept):
            pre_computed[concept] = getattr(sample, f"get_{concept}")(points=points, image_shape=img.shape[:2])

    return pre_computed

@timeit
def get_targets(sample, img, points, task_concepts, targets_config, augment=False, new_background=None):

    pre_computed = pre_compute(sample, img, points, task_concepts)

    targets = []
    for tc in targets_config:
        if tc["task"] in pre_computed:
            if tc["name"] in pre_computed[tc["task"]]:
                target = pre_computed[tc["task"]][tc["name"]]
            else:
                target = np.zeros(shape=img.shape[:2] + (tc["channels"],), dtype=tc["dtype"])
        elif tc["task"] == "encoder":
            if tc["name"] == "identity":
                target = img.copy()
            elif tc["name"] == "identity_bw":
                target = img.mean(axis=2)
        elif tc["task"] == "hierarchy":
            target = sample.get_hierarchy(points=points, notions=tc["concepts"])

        else:
            target = getattr(sample, f'get_{tc["name"]}')(points=points)

        targets.append(target)

    return targets


def main(
    directory="/dev/shm/soleil_proxima2a",
    # directory="/nfs/data2/Martin/Research/murko/manually_segmented_images/json/spine/soleil_proxima2a",
):
    import subprocess
    targets_config, task_concepts = get_candidates()
    print(f'targets_config ({len(targets_config)}):')
    for tc in targets_config:
        print(tc)
    print()
    print(f"task_concepts ({len(task_concepts)}):")
    for task_concept in task_concepts:
        print(task_concept)
    print()

    annotations = subprocess.getoutput(f'find {directory} -iname "*.json"').split("\n")

    print(f"number of annotations {len(annotations)}")
    dl = JsonDataset(
        annotations,
        targets_config=targets_config,
        task_concepts=task_concepts,
    )
    
    return dl

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument(
        "-d",
        "--directory",
        # default="/nfs/data2/Martin/Research/murko/manually_segmented_images/json/spine/soleil_proxima2a",
        default="/dev/shm/soleil_proxima2a",
        type=str,
        help="directory",
    )
    args = parser.parse_args()
    print(args)
    main(directory=args.directory)
