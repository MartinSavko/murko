#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import os
import json
import numpy as np
import cv2 as cv
from skimage.transform import resize

from objects_of_interest import get_objects_of_interest, update_maps
from regionprops import Regionprops
from config import notion_importance, keypoints, keypoint_labels

def get_label_mask_from_points(oois, labels, points=None):
    image_shape = oois["image_shape"]
    label_mask = np.zeros(image_shape, dtype=np.uint8)

    if "any" in labels:
        labels = list(set(oois["labels"]))

    label_list = oois["labels"]
    label_indices = oois["indices"]

    if points is None:
        points = oois["points"]
    else:
        assert len(oois["points"]) == len(points)

    for label in labels:
        if label not in label_list:
            continue

        for i_start, i_end in [
            label_indices[k] for k, item in enumerate(label_list) if item == label
        ]:
            ps = points[i_start:i_end]
            if len(ps) < 3:
                continue
            polygon = ps * image_shape
            mask = get_mask_from_polygon(polygon, image_shape)
            label_mask = np.logical_or(label_mask == 1, mask == 1)
    return label_mask


def get_hierarchy_from_oois(
    oois,
    points=None,
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "pin",
        "stem",
        "foreground",
        "background",
    ],
    notion_importance=notion_importance,
):
    notions.sort(key=lambda x: -notion_importance[x])
    notion_values = np.array([notion_importance[notion] for notion in notions])

    image_shape = oois["image_shape"]
    hierarchical_target = np.zeros(tuple(image_shape) + (len(notions),))

    for label in oois["labels"]:
        label_mask = get_label_mask_from_points(oois, [label], points=points)
        if label in notions:
            i = notions.index(label)
        elif label != "background":
            i = notions.index("foreground")
        hierarchical_target[:, :, i] = np.logical_or(
            hierarchical_target[:, :, i], label_mask
        )

    hierarchical_target /= notion_importance
    hierarchical_mask = np.argmax(hierarchical_target, axis=2)
    return hierarchical_mask


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def get_flipped_image(image, axis):
    flipped_image = flip_axis(img, axis)
    return flipped_image


def get_flipped_img_and_points(img, points):
    axis = random.choice([0, 1])
    fimg = get_flipped_image(img, axis)
    fpoints = points[:, :]
    fpoints[:, axis] = img.shape[axis] - points[:, axis]
    return fimg, fpoints


def get_transposed_image(image):
    new_axes_order = (1, 0) + tuple(range(2, len(image.shape)))
    transposed_image = np.transpose(imag, new_axes_order)
    return transposed_image


def get_transposed_img_and_points(img, points):
    timg = get_transposed_image(img)
    tpoints = points[:, ::-1]
    return timg, tpoints


def get_transformed_image(
    img, transformation, output_shape=None, doer="ski", cval=-1, mode="constant"
):
    if output_shape is None:
        output_shape = img.shape
    if doer == "ski":
        transformed_image = ski.transform.warp(
            img, transformation, output_shape=output_shape, cval=cval, mode=mode
        )
    elif doer == "cv":
        if mode == "constant":
            borderMode = cv.BORDER_CONSTANT
        elif mode == "edge":
            borderMode = cv.BORDER_REPLICATE
        transformed_image = cv.warpAffine(
            img,
            transformation._inv_matrix[:2, :],
            output_shape[::-1],
            borderValue=[cval] * 3,
            borderMode=borderMode,
        )
    return transformed_image


def get_transformed_points(points, transformation_matrix):
    points = points[:, [1, 0, 2]]
    transformed_points = np.dot(transformation_matrix, points.T).T
    transformed_points = transformed_points[:, [1, 0, 2]]
    return transformed_points


def get_transformed_img_and_points(img, points):
    transformation = get_random_transformation()
    timage = get_transformed_image(img, transformation)
    tpoints = get_transformed_points(points, transformation._inv_matrix)
    return timage, tpoints


# zoom_factor=0.25,
# shift_factor=0.25,
# shear_factor=45,
# default_transform_gang=[0, 0, 0, 0, 1, 1],
def get_random_transformation(
    rotation_range=np.pi,
    scale_range=0.5,
    translation_range=0.25,
    shear_range=0.5 * np.pi,
    img_shape=np.array((1200, 1600)),
    rotation_center="random",
):
    if rotation_center == "random":
        r_center = np.random.random(size=2) * img_shape
    else:
        r_center = img_shape / 2

    shift_c = ski.transform.AffineTransform(translation=-r_center)
    shift_invc = ski.transform.AffineTransform(translation=+r_center)

    rotation = (np.random.rand() - 0.5) * rotation_range
    scale = 1 + (np.random.random(size=2) - 0.5) * scale_range
    shear = (np.random.random(size=2) - 0.5) * shear_range
    translation = [
        0,
        0,
    ]  # (np.random.random(size=2) - 0.5) * translation_range * img_shape

    print(f"rotation {rotation}, rotation_center {r_center}")
    print(f"scale {scale}")
    print(f"shear {shear}")
    print(f"translation {translation}")

    t_rotation = ski.transform.AffineTransform(rotation=rotation)
    t_scale = ski.transform.AffineTransform(scale=scale)
    t_shear = ski.transform.AffineTransform(shear=shear)
    t_translation = ski.transform.AffineTransform(translation=translation)

    # random_transformation = ski.transform.AffineTransform(
    # scale=scale, rotation=rotation, shear=shear, translation=translation
    # )

    random_transformation = (
        shift_c + t_rotation + t_scale + t_shear + shift_invc + t_translation
    )

    return random_transformation


class Sample:
    def __init__(self, json_file, not_to_keep=["masks", "properties"]):
        if os.path.isfile(json_file):
            json_file = load_json(json_file)

        self.oois = get_objects_of_interest(json_file)
        self.image_path = self.oois["image_path"]
        self.points = self.oois["points"]
        self.indices = self.oois["indices"]
        self.labels = self.oois["labels"]
        self.fractional = self.oois["fractional"]
        for key in not_to_keep:
            del self.oois[key]

    def get_target(self, head, img, points):
        pass

    def get_image(self):
        return self.oois["image"].copy()

    def get_points(self):
        return self.oois.copy()

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

    def _get_properties(self, points=None, image_shape=None):
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
            if self.fractional:
                ps *= image_shape
            props = Regionprops(ps, image_shape=image_shape)
            properties.append(props)
        return properties

    def get_masks(self, points=None, image_shape=None):
        masks = self._get_maps(points, image_shape, kind="mask", method="logical_or")
        return masks

    def get_distance_transform(self, points=None, image_shape=None):
        distance_transform = self._get_maps(
            points, image_shape, kind="distance_transform", method="max"
        )
        return distance_transform

    def get_centerness(self, points=None, image_shape=None):
        centerness = self._get_maps(
            points, image_shape, kind="centerness", method="min"
        )
        return centerness

    def get_bbox_mask(self, points=None, image_shape=None):
        bbox_mask = self._get_maps(
            points, image_shape, kind="bbox_mask", method="logical_or"
        )
        return bbox_mask

    def get_bbox_ltrb(self, points=None, image_shape=None):
        bbox_ltrb = self._get_maps(
            points, image_shape, kind="bbox_ltrb", method="logical_or"
        )
        return bbox_ltrb

    def get_ellipse_mask(self, points=None, image_shape=None):
        ellipse_mask = self._get_maps(
            points, image_shape, kind="ellipse_mask", method="logical_or"
        )
        return ellipse_mask

    def get_min_rectangle_mask(self, points=None, image_shape=None):
        min_rectangle_mask = self._get_maps(
            points, image_shape, kind="min_rectangle_mask", method="logical_or"
        )
        return min_rectangle_mask

    def get_most_likely_click(self):
        pass

    def get_keypoints(self):
        pass

    def get_voronoi(
        self,
        keypoints,  # most_likely_click, aoi_start, aoi_end, aoi_top, aoi_bottom, start_possible, origin
        image_shape,
    ):
        """http://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/"""

        cv.Subdiv2D((0, 0, image_shape[1], image_shape[0]))
        for key, p in keypoints.items():
            if p is not None:
                subdiv.insert(p)
        facets, centers = subdiv.getVoronoiFacetList([])
        voronoi = np.zeros(image_shape, dtype=np.int8)
        i = 0
        for key, p in keypoints:
            label = keypoint_labels[key]
            if p is not None:
                facet = facets[i]
            else:
                continue
            polygon = []
            for f in facet:
                polygon.append(f)
            cv.fillPoly(voronoi, np.array(polygon, dtype=np.int32), label)
        return voronoi

    def transform(self, final_img_size, new_background=None):
        (
            do_flip,
            do_transpose,
            do_transform,
            do_swap_backgrounds,
            do_black_and_white,
            do_random_brightness,
            do_random_channel_shift,
        ) = self.get_augment_control()

        img = self.get_image()
        points = self.get_points()

        if do_transpose is True:
            img, points = get_transposed_img_and_points(img, points)

        if do_flip is True:
            img, points = get_flipped_img_and_points(img, points)

        if do_transform is True:
            img, points = get_transformed_img_and_points(img, points)

        masks = self.get_masks(points, img.shape[:2])

        if (
            new_background is not None
            and do_swap_backgrounds
            and "background" not in self.image_path
            and "foreground" in masks
        ):
            img = self.swap_backgrounds(img, masks["foreground"], new_background)

        if size_differs(img.shape[:2], final_img_size):
            resize_factor = np.array(final_img_size) / np.array(img.shape[:2])
            img = resize(img, final_img_size, anti_aliasing=True)
            if not self.fractional:
                points = points * resize_factor

        if do_random_brightness is True:
            img = image.random_brightness(img, [0.75, 1.25]) / 255.0

        if do_random_channel_shift is True:
            img = image.random_channel_shift(img, 0.5, channel_axis=2)

        if do_black_and_white:
            img_bw = img.mean(axis=2)
            img = np.stack([img_bw] * 3, axis=2)

        return img, points

    def swap_backgrounds(self, img, foreground_mask, new_background):
        if size_differs(img.shape[:2], new_background.shape[:2]):
            new_background = resize(new_background, img.shape[:2], anti_aliasing=True)
        img[foreground_mask == 0] = new_background[foreground_mask == 0]
        return img

    def get_augment_control(
        self,
        threshold=0.5,
        transform=True,
        transpose=True,
        flip=True,
        swap_backgrounds=True,
        black_and_white=True,
        random_brightness=True,
        random_channel_shift=False,
        verbose=False,
    ):

        do_flip = False
        do_transpose = False
        do_transform = False
        do_swap_backgrounds = False
        do_black_and_white = False
        do_random_brightness = False
        do_random_channel_shift = False
        if transform and self.transform and random.random() < threshold:
            do_transform = True

        if transpose and random.random() < threshold:
            final_img_size = img_size[::-1]
            do_transpose = True
            if self.flip and random.random() < threshold:
                do_flip = True
        else:
            if flip and random.random() < threshold:
                do_flip = True

        if swap_backgrounds and random.random() < threshold / 2:
            do_swap_backgrounds = True

        if black_and_white and random.random() < threshold / 2:
            do_black_and_white = True

        if random_brightness and random.random() < threshold / 2:
            do_random_brightness = True

        if (
            random_channel_shift
            and not do_black_and_white
            and random.random() < threshold / 2
        ):
            do_random_channel_shift = True

        if verbose:
            print(f"do_transform: {do_transform}")
            print(f"do_transpose: {do_transpose}")
            print(f"do_flip: {do_flip}")
            print(f"do_swap_backgrounds: {do_swap_backgrounds}")
            print(f"do_black_and_white: {do_black_and_white}")
            print(f"do_random_brightness: {do_random_brightness}")
            print(f"do_random_channel_shift: {do_random_channel_shift}")
        return (
            do_flip,
            do_transpose,
            do_transform,
            do_swap_backgrounds,
            do_black_and_white,
            do_random_brightness,
            do_random_channel_shift,
        )
