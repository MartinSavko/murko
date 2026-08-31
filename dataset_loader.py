#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import time
import copy
import copy
import math
import random
import numpy as np
import traceback
import pylab

from keras.utils import Sequence, PyDataset, to_categorical
from sample import Sample
from candidates import get_candidates

from utils import (
    label2rgb,
)

from config import luts

def timeit(func):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        print("func:%r took: %2.8f sec" % (func.__name__, te - ts))
        return result

    return timed


# from config import targets_config


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


@timeit
def load_samples_from_annotations(annotations, preferred_image_size=None):
    samples = [
        Sample(item, preferred_image_size=preferred_image_size) for item in annotations
    ]
    return samples


class JsonDataset(PyDataset):

    def __init__(
        self,
        annotations,
        targets_config,
        batch_size=1,
        dynamic_batch_size=False,
        number_batch_size_scales=32,
        img_size=(256, 256),
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
        workers=32,
        use_multiprocessing=False,
        max_queue_size=128,
    ):

        self.annotations = annotations
        self.targets_config = targets_config
        self.task_concepts = list(set([item["task"] for item in self.targets_config]))

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

        self.load_samples(preferred_image_size=self.img_size)

        self.artificial_size_increase = artificial_size_increase
        if self.artificial_size_increase > 1:
            self.samples *= self.artificial_size_increase

        self.nsamples = len(self.samples)

        if self.swap_backgrounds:
            # self.backgrounds = [
            #     sample
            #     for sample in self.samples
            #     if "background" in sample.image_path.lower()
            # ]
            self.backgrounds = [
                sample for sample in self.samples if "foreground" not in sample.labels
            ]

        self.shuffle_at_0 = shuffle_at_0
        self.target = target

        self.require_transpose = False
        self.disallow_transpose = True
        self.verbose = verbose

        super().__init__(
            workers=workers,
            use_multiprocessing=use_multiprocessing,
            max_queue_size=max_queue_size,
        )

    def load_samples(self, preferred_image_size=None):
        self.samples = load_samples_from_annotations(
            self.annotations, preferred_image_size=preferred_image_size
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
            batch = self.samples[start_index:end_index]

        # transpose should probably better be decided on the batch level
        if self.augment and random.random() < 0.5:
            self.require_transpose = True
            self.disallow_transpose = False
            img_size = img_size[::-1]
        else:
            self.require_transpose = False
            self.disallow_transpose = True

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
                    y[k][j] = target

        if self.target:
            if len(y) == 1:
                y = y[0]
            item = x, tuple(y)
        else:
            item = x

        return item

    # @timeit
    def get_image_and_targets(self, sample, img_size, new_background=None):

        if self.augment and self.swap_backgrounds:
            new_background = random.choice(self.backgrounds).get_image()

        img, points = sample.get_image_and_points(
            img_size=img_size,
            augment=self.augment,
            new_background=new_background,
            require_transpose=self.require_transpose,
            disallow_transpose=self.disallow_transpose,
            verbose=self.verbose,
        )

        if not self.target:
            return img, None

        # print("points\n", points)
        targets = get_targets(
            sample,
            img,
            points,
            self.task_concepts,
            self.targets_config,
            self.augment,
            new_background=new_background,
        )

        return img, targets


def pre_computable(concept):
    return (
        "binary_segment" in concept
        or "distance_transform" in concept
        or "centerness" in concept
    )


# @timeit
def pre_compute(sample, img, points, task_concepts):
    pre_computed = {}
    for concept in task_concepts:
        if pre_computable(concept):
            pre_computed[concept] = getattr(sample, f"get_{concept}")(
                points=points, image_shape=img.shape[:2]
            )

    return pre_computed


# @timeit
def get_targets(
    sample,
    img,
    points,
    task_concepts,
    targets_config,
    augment=False,
    new_background=None,
):

    pre_computed = pre_compute(sample, img, points, task_concepts)

    masks = None
    if "binary_segment" in pre_computed:
        masks = pre_computed["binary_segment"]

    targets = []
    for tc in targets_config:
        if tc["task"] in pre_computed:
            if tc["name"] in pre_computed[tc["task"]]:
                target = pre_computed[tc["task"]][tc["name"]]
            else:
                target = np.zeros(
                    shape=img.shape[:2] + (tc["channels"],), dtype=tc["dtype"]
                )
        elif tc["task"] == "encoder":
            if tc["name"] == "identity":
                target = img
            elif tc["name"] == "identity_bw":
                target = img.mean(axis=2, keepdims=True)
        elif tc["task"] == "hierarchy":
            flat_hierarchy = sample.get_flat_hierarchy(
                points=points,
                image_shape=img.shape[:2],
                masks=masks,
                notions=tc["concepts"],
            )
            target = to_categorical(flat_hierarchy, num_classes=len(tc["concepts"]))
        elif tc["task"] == "global_classification":
            target = sample.get_global_classification_target(
                points=points,
                image_shape=img.shape[:2],
                masks=masks,
                notions=tc["concepts"],
                focus=masks[tc["focus"]],
            )
        else:
            target = getattr(sample, f'get_{tc["name"]}')(points=points)

        if target.shape != img.shape[:2] + (tc["channels"],):
            target_size = np.prod(img.shape[:2]) * tc["channels"]
            try:
                if np.prod(target.shape) == target_size:
                    target = np.reshape(target, img.shape[:2] + (tc["channels"],))
                elif np.prod(target.shape) * tc["channels"] == target_size:
                    target = [target] * tc["channels"]
            except:
                traceback.print_exc()

        targets.append(target)

    return targets


def plot_image_and_targets(
    image,
    targets,
    targets_config,
    figsize=(24, 16),
    threshold=None,
    verbose=False,
    path=None,
    original_image=None,
    model_designation="_",
    save=False,
    close=False,
):

    N = 1 + len(targets)
    if original_image is not None: N += 1
    rows = np.floor(np.sqrt(N))
    cols = np.ceil(N / rows)
    assert rows * cols >= N

    fig, axes = pylab.subplots(int(rows), int(cols), figsize=figsize)
    axs = axes.flatten()
    if type(path) is str:
        fig.suptitle(
            path.replace(
                "/nfs/data2/Martin/Research/murko/manually_segmented_images/json/", ""
            ),
            fontsize=24,
        )

    l = 0
    if original_image is not None:
        axs[l].imshow(original_image)
        axs[l].set_title("pristine input")
        l += 1

    if len(image.shape) == 4:
        image = image[0]

    axs[l].imshow(image)
    axs[l].set_title("Input image")
    l += 1
    for k, (target, config) in enumerate(zip(targets, targets_config)):
        if len(target.shape) == 4:
            target = target[0]
        if config["channels"] in [1, 3] and config["task"] != "hierarchy":
            if (
                threshold is not None
                and config["channels"] == 1
                and config["name"] != "identity_bw"
                and "binary_segment" in config["task"]
            ):
                if verbose:
                    print(f"config\n{config}")
                    print(f"target", target.shape, target.dtype)
                target = (target >= threshold).astype("uint8")
                target = label2rgb(target[:,:,0], luts[f'{config["name"]}_{config["task"]}'])

            elif "binary_segment" in config["task"]:
                if verbose:
                    print(f"config\n{config}")
                    print(f"target", target.shape, target.dtype)
                target = label2rgb(target[:,:,0].astype("uint8"), luts[f'{config["name"]}_{config["task"]}'])

            if config["name"] == "identity_bw":
                axs[k + l].imshow(target, cmap="gray")
            else:
                axs[k + l].imshow(target)
        elif config["task"] == "hierarchy":
            label = np.argmax(target, axis=2).astype("uint8")
            if verbose:
                print(f"config\n{config}")
                print(f"label", label.shape, label.dtype)
            axs[k + l].imshow(label2rgb(label, luts[config["name"]]))

        title = f'{config["name"]} {config["task"]}'.replace("hierarchy_", "").replace("area_of_interest", "aoi").replace("distance_transform", "dt").replace("binary_segment", "")
        axs[k + l].set_title(title)

    for ax in axs:
        ax.set_axis_off()

    if save and path is not None:
        report_png = path.replace(".json", f"{model_designation}_results.png")
        if verbose:
            print(f"saving report {report_png}")
        pylab.savefig(report_png)
    if close:
        pylab.close(fig)


def main(
    directory="/nfs/data2/Martin/Research/murko/manually_segmented_images/json/spine/soleil_proxima2a",
):
    import subprocess

    targets_config, task_concepts = get_candidates()
    print(f"targets_config ({len(targets_config)}):")
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
        targets_config,
    )

    return dl


if __name__ == "__main__":
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
