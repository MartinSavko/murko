#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import os
import json
import numpy as np

from show_annotations import (
    load_json,
    cvRegionprops,
    get_objects_of_interest,
    get_hierarchy_from_oois,
    get_primary_masks,
    get_distance_transforms,
)
from dataset_loader import (
    get_transposed_img_and_points,
    get_flipped_img_and_points,
    get_transformed_img_and_points,
    resize,
)


def merge_maps(map1, map2, method):
    if "logical" in method:
        mmap = getattr(np, method)(map1, map2)
    else:
        mmap = getattr(np, method)(np.stack([map1, map2], axis=0))
    return mmap


def update_maps(maps, label, map, method="logical_or"):
    maps[label] = merge_maps(maps[label], map, method) if label in maps else map
    return maps


class sample:

    def __init__(
        self,
        json_file,
    ):
        if os.path.isfile(json_file):
            json_file = load_json(json_file)

        self.oois = get_objects_of_interest(json_file)
        self.indices = self.oois["indices"]
        self.labels = self.oois["labels"]
        self.fractional = self.oois["fractional"]

    def get_hierarchy(
        self,
        points=None,
        notions=[
            "crystal",
            "loop_inside",
            "loop",
            "stem",
            "pin",
            "foreground",
            "background",
        ],
    ):
        hierarchy = get_hierarchy_from_oois(self.oois, points=points, notions=notions)
        return hierarchy

    def _get_maps(
        self, points=None, image_shape=None, kind="mask", method="logical_or", **kwargs
    ):

        _maps = {}
        properties = self._get_properties(points, image_shape)
        for k, label in enumerate(self.labels):
            _map = getattr(properties[k], f"get_{kind}")(**kwargs)
            update_maps(_maps, label, _map, method=method, type=type)
        return _maps

    def _get_properties(
        self,
        points=None,
        image_shape=None,
    ):
        if points is None:
            points = self.oois["points"]
        if image_shape is None:
            image_shape = self.oois["image_shape"]

        properties = []
        for k, label in enumerate(self.labels):
            i_start, i_end = self.indices[k]
            ps = points[i_start:i_end]
            if len(ps) < 3:
                continue
            if fractional:
                ps *= image_shape
            props = cvRegionprops(ps, image_shape=image_shape)
            properties.append(props)
        return properties

    def get_mask(
        self,
        points=None,
        image_shape=None,
    ):
        mask = self._get_maps(
            points, image_shape, kind="mask", method="logical_or"
        )
        return mask

    def get_distance_transform(
        self,
        points=None,
        image_shape=None,
    ):
        distance_transform = self._get_maps(
            points, image_shape, kind="distance_transform", method="max"
        )
        return distance_transform

    def get_centerness(
        self,
        points=None,
        image_shape=None,
    ):
        centerness = self._get_maps(
            points, image_shape, kind="centerness", method="min"
        )
        return centerness
    
    def get_bbox_mask(
        self,
        points=None,
        image_shape=None,
    ):
        bbox_mask = self._get_maps(
            points, image_shape, kind="bbox_mask", method="logical_or"
        )
        return bbox_mask

    def get_bbox_ltrb(
        self,
        points=None,
        image_shape=None,
    ):
        bbox_ltrb = self._get_maps(
            points, image_shape, kind="bbox_ltrb", method="logical_or"
        )
        return bbox_ltrb

    def get_ellipse_mask(
        self,
        points=None,
        image_shape=None,
    ):
        ellipse_mask = self._get_maps(
            points, image_shape, kind="ellipse_mask", method="logical_or"
        )
        return ellipse_mask

    def get_min_rectangle_mask(
        self,
        points=None,
        image_shape=None,
    ):
        min_rectangle_mask = self._get_maps(
            points, image_shape, kind="min_rectangle_mask", method="logical_or"
        )
        return min_rectangle_mask

    
      
    def transform(self, final_img_size):
        (
            do_flip,
            do_transpose,
            do_transform,
            do_swap_backgrounds,
            do_black_and_white,
            do_random_brightness,
            do_random_channel_shift,
        ) = self.get_transform_control()

        img = self.oois["image"]
        img_path = self.oois["image_path"]
        points = self.oois["points"]
        fractional = self.oois["fractional"]

        if do_transpose is True:
            img, points = get_transposed_img_and_points(img, points)

        if do_flip is True:
            img, points = get_flipped_img_and_points(img, points)

        if do_transform is True:
            img, points = get_transformed_img_and_points(img, points)

        masks = get_primary_masks(
            points, indices, labels, img.shape[:2], fractional=fractional
        )

        if (
            do_swap_backgrounds
            and "background" not in img_path
            and "foreground" in masks
        ):
            img = self.swap_backgrounds(img, masks["foreground"])

        if size_differs(img.shape[:2], final_img_size):
            resize_factor = np.array(final_img_size) / np.array(img.shape[:2])
            img = resize(img, final_img_size, anti_aliasing=True)
            if not fractional:
                points = points * resize_factor

        if do_random_brightness is True:
            img = image.random_brightness(img, [0.75, 1.25]) / 255.0

        if do_random_channel_shift is True:
            img = image.random_channel_shift(img, 0.5, channel_axis=2)

        if do_black_and_white:
            img_bw = img.mean(axis=2)
            img = np.stack([img_bw] * 3, axis=2)

        return img, points

    def swap_backgrounds(self, img, foreground_mask):
        new_background = random.choice(self.backgrounds)["image"]
        if size_differs(img.shape[:2], new_background.shape[:2]):
            new_background = resize(new_background, img.shape[:2], anti_aliasing=True)
        img[foreground_mask == 0] = new_background[foreground_mask == 0]
        return img

    def get_transform_control(self):
        do_flip = False
        do_transpose = False
        do_transform = False
        do_swap_backgrounds = False
        do_black_and_white = False
        do_random_brightness = False
        do_random_channel_shift = False
        if self.transform and random.random() < self.threshold:
            do_transform = True
            if self.verbose:
                print("do_transform")
        if self.transpose and random.random() < self.threshold:
            final_img_size = img_size[::-1]
            do_transpose = True
            if self.verbose:
                print("do_transpose")
            if self.flip and random.random() < self.threshold:
                do_flip = True
                if self.verbose:
                    print("do_flip")
        else:
            if self.flip and random.random() < self.threshold:
                do_flip = True
                if self.verbose:
                    print("do_flip")
        if self.swap_backgrounds and random.random() < self.threshold / 2:
            do_swap_backgrounds = True
            if self.verbose:
                print("do_swap_backgrounds")
        if self.black_and_white and random.random() < self.threshold / 2:
            do_black_and_white = True
            if self.verbose:
                print("do_black_and_white")
        if self.random_brightness and random.random() < self.threshold / 2:
            do_random_brightness = True
            if self.verbose:
                print("do_random_brightness")
        if (
            not do_black_and_white
            and self.random_channel_shift
            and random.random() < self.threshold / 2
        ):
            do_random_channel_shift = True
            if self.verbose:
                print("do_random_channel_shift")
        return (
            do_flip,
            do_transpose,
            do_transform,
            do_swap_backgrounds,
            do_black_and_white,
            do_random_brightness,
            do_random_channel_shift,
        )
