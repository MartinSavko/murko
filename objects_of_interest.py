#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

# https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html

import os
import json
import traceback
import numpy as np
import cv2 as cv
import imageio

try:
    import labelme
except:
    labelme = None
    
from config import additional_labels, notion_importance, keypoints, keypoint_labels
from regionprops import (
    Regionprops,
    adjust_points,
)

from utils import get_valid_image_and_points


def load_json(
    fname="/nfs/data2/Martin/Research/murko/manually_segmented_images/json/spine/dls_i04/6116020_fullscreen-30086648_201.40800000000002.json",
):
    f = json.load(open(fname, "rb"))
    return f


def get_image(json_file, json_path=None):
    imageData = json_file.get("imageData")
    if imageData is not None:
        image = labelme.utils.img_b64_to_arr(imageData) / 255.0
    else:
        if json_path is not None:
            image_path = os.path.join(
                os.path.dirname(json_path), json_file.get("imagePath")
            )
        else:
            image_path = json_file.get("imagePath")
        image = imageio.imread(image_path)

        if image.dtype == "uint8":
            image = image / 255.0

    if image.shape[0] == 3:
        image = np.moveaxis(image, 0, -1)

    if len(image.shape) == 2:
        image = np.stack([image] * 3, axis=2)

    return image.astype("float32")


def get_image_shape(json_file):
    image_shape = np.array((json_file.get("imageHeight"), json_file.get("imageWidth")))
    return image_shape


def get_image_path(json_file):
    image_path = json_file["imagePath"]
    return image_path


def get_shapes(json_file):
    shapes = json_file.get("shapes")
    return shapes


def get_label_points(label, labels, indices, points, fractional, image_shape):
    label_points = None
    idx = labels.index(label) if label in labels else None
    if idx is not None:
        i_start, i_end = indices[idx]
        label_points = points[i_start:i_end]
        if fractional:
            label_points *= image_shape
    return label_points


def remove_ooi(label, labels, indices, points, properties):
    idx = labels.index(label) if label in labels else None
    if idx is not None:
        # start, end = indices[idx]
        # points = np.delete(points, np.s_[start:end])
        del labels[idx]
        del indices[idx]
        del properties[idx]
    return labels, indices, points, properties


def add_ooi(ooi, label, labels, indices, points, properties, image_shape):
    if indices:
        i_start = indices[-1][-1]
    else:
        i_start = 0
    i_end = i_start + ooi.shape[0]
    points = np.vstack([points, ooi]) if len(points) else ooi
    indices.append((i_start, i_end))
    labels.append(label)
    properties.append(Regionprops(ooi, image_shape=image_shape))

    return labels, indices, points, properties


def get_masks(labels, indices, points, properties, image_shape, fractional=False):

    masks = {}

    for k, label in enumerate(labels):
        i_start, i_end = indices[k]
        ps = points[i_start:i_end]
        if len(ps) < 3:
            continue
        if fractional:
            ps *= image_shape

        mask = properties[k].get_mask(image_shape=image_shape)
        masks = update_maps(masks, label, mask)
        if label != "background":
            masks = update_maps(masks, "foreground", mask)

        if label in ["crystal", "loop"]:
            masks = update_maps(masks, "area_of_interest", mask)

        if label in ["loop", "stem"]:
            masks = update_maps(masks, "support", mask)

        if label in ["crystal", "loop", "stem"]:
            masks = update_maps(masks, "explorable", mask)

    if "pin" in masks:
        for concept in ["stem", "explorable", "support"]:
            if concept in masks:
                masks[concept][masks["pin"].astype(bool)] = 0

    if "support" in masks:
        masks["plastic"] = masks["support"].copy()
        if "loop_inside" in masks:
            masks["plastic"][masks["loop_inside"].astype(bool)] = 0

    if "background" in masks:
        for body in ["foreground", "crystal", "area_of_interest"]:
            aether = "aether" if body == "foreground" else f'{body}_aether'
            masks[aether] = masks["background"].copy()
            # if body in masks:
            #     masks[aether][masks[body].astype(bool)] = 0

    return masks


def merge_maps(map1, map2, method):

    if "logical" in method:
        mmap = getattr(np, method)(map1, map2)
    else:
        # this may be useful if we are interested in minimum or maximum
        # e.g. when merging distance transforms, point regions etc.
        # mask = np.logical_and(map1>0, map2>0)
        stack = np.stack([map1, map2], axis=0)
        mmap = getattr(np, method)(stack, axis=0)  # , where=mask)
        # mmap = map1 + map2
        # m = mmap.max()
        # if m > 0:
        # mmap /= m
    return mmap


def update_maps(maps, label, _map, method="logical_or"):
    try:
        maps[label] = merge_maps(maps[label], _map, method) if label in maps else _map
    except:
        traceback.print_exc()
    return maps


def add_derived_notions(
    labels,
    indices,
    points,
    properties,
    image_shape,
    fractional=False,
    derived_notions=[
        "foreground",
        "area_of_interest",
        "support",
        "explorable",
        "plastic",
        "stem",
        "aether",
        "area_of_interest_aether",
        "crystal_aether",
    ],
):

    masks = get_masks(labels, indices, points, properties, image_shape)

    if "stem" in derived_notions and "stem" in masks:
        labels, indices, points, properties = remove_ooi(
            "stem", labels, indices, points, properties
        )

    for notion in derived_notions:
        if notion in masks:
            contours, h = cv.findContours(
                masks[notion].astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            for contour in contours:
                shape = contour.shape
                new_shape = (shape[0], shape[2])
                ooi = contour.reshape(new_shape)[:, ::-1]
                if fractional:
                    ooi /= image_shape

                labels, indices, points, properties = add_ooi(
                    ooi, notion, labels, indices, points, properties, image_shape
                )

    return labels, indices, points, properties, masks

def get_objects_of_interest(
    json_file,
    fractional=False,
    unit_square=np.array([[0, 0], [1, 0], [1, 1], [0, 1]]),
    json_path=None,
    verbose=False,
):
    if type(json_file) is str and os.path.isfile(json_file):
        json_path = json_file
        json_file = load_json(json_file)
        realpath = os.path.realpath(json_path)
    else:
        json_path = None
        realpath = None

    image = get_image(json_file, json_path=json_path)

    image_shape = get_image_shape(json_file)
    image_path = get_image_path(json_file)

    labels, indices, points, properties = [], [], [], []

    support_type = None

    for shape in get_shapes(json_file):
        raw_label = shape["label"]
        if raw_label in additional_labels:
            label = additional_labels[raw_label]
        else:
            label = raw_label

        if label == "loop":
            if raw_label == "loop":
                support_type = "standard"
            elif "mt" in raw_label:
                support_type = "mitegen"
            elif "cd" in raw_label:
                support_type = "crystal_direct"

        ooi = np.array(shape["points"])
        ooi = ooi[
            :, ::-1  # swap x and y (labelme uses [h, v] convention, we use [v, h]
        ]

        if shape["shape_type"] == "rectangle" and ooi.shape[0] == 2:
            ooi = unit_square * np.abs(ooi[0] - ooi[1])

        if fractional:
            ooi /= image_shape

        labels, indices, points, properties = add_ooi(
            ooi, label, labels, indices, points, properties, image_shape
        )

    if "background" not in labels:
        if not fractional:
            ooi = unit_square[:] * image_shape

        labels, indices, points, properties = add_ooi(
            ooi, "background", labels, indices, points, properties, image_shape
        )

    labels, indices, points, properties, masks = add_derived_notions(
        labels, indices, points, properties, image_shape, fractional=fractional
    )

    background_points = get_label_points("background", labels, indices, points, fractional, image_shape)

    image, points, valid_shift = get_valid_image_and_points(background_points, image, points)

    image_shape = image.shape
    points = adjust_points(points, image_shape)

    objects_of_interest = {
        "realpath": realpath,
        "json_path": json_path,
        "image": image,
        "image_shape": image_shape,
        "image_path": image_path,
        "fractional": fractional,
        "labels": labels,
        "indices": indices,
        "points": points.astype("float32"),
        "properties": properties,
        "masks": masks,
        "support_type": support_type,
    }

    return objects_of_interest
