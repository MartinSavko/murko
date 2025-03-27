#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import time
import json
import numpy as np
import skimage as ski
from skimage.draw import polygon2mask
import scipy.ndimage as ndi
import cv2 as cv
import imageio

import pylab
import matplotlib.patches
import seaborn as sns

from labelme import utils
import time
import copy
import traceback

# from utils import get_extreme_point

xkcd_colors_that_i_like = [
    "pale purple",
    "coral",
    "moss green",
    "windows blue",
    "amber",
    "greyish",
    "faded green",
    "dusty purple",
    "crimson",
    "custard",
    "orangeish",
    "dusk blue",
    "ugly purple",
    "carmine",
    "faded blue",
    "dark aquamarine",
    "cool grey",
]

colors_for_labels = {
    "background": "dark green",
    "degraded_protein": "chartreuse",
    "crystal_cluster": "greenish yellow",
    "air_bubble": "light aqua",
    "precipitate": "pale yellow",
    "crystalline_precipitate": "pale yellow",
    "crystal_shower": "olive drab",
    "feather-like": "greyish blue",
    "urchin": "light olive",
    "not_background": "greyish",
    "pin": "dusk blue",
    "stem": "crimson",
    "loop": "faded green",
    "loop_inside": "moss green",
    "ice": "ice",
    "dust": "cool grey",
    "capillary": "faded blue",
    "crystal": "banana yellow",
    "drop": "orangeish",
    "support": "dusk blue",
    "support_filled": "orangeish",
    "area_of_interest": "dark green",
    "user_click": "banana yellow",
    "extreme": "dark aquamarine",
    "start_likely": "coral",
    "start_possible": "crimson",
    "most_likely_point": "orangeish",
    "extreme_point": "carmine",
    "start_likely_point": "dusk blue",
    "start_possible_point": "cool grey",
}

additional_labels = {
    "cd_loop": "loop",
    "cd_stem": "stem",
    "loop_cd": "loop",
    "stem_cd": "stem",
    "loop_mt": "loop",
    "stem_mt": "stem",
    "mt_loop": "loop",
    "mt_stem": "stem",
}

# 8 + 1 + 2 + 8 + 8 + 4 = 31
targets = {
    # 8 binary segmentations
    "crystal": {"type": "binary_segmentation"},
    "loop_inside": {"type": "binary_segmentation"},
    "loop": {"type": "binary_segmentation"},
    "stem": {"type": "binary_segmentation"},
    "pin": {"type": "binary_segmentation"},
    "foreground": {"type": "binary_segmentation"},
    "ice": {"type": "binary_segmentation"},
    "area_of_interest": {"type": "binary_segmentation"},
    "support": {"type": "binary_segmentation"},
    # 1 categorical segmentation
    "hierarchy": {"type": "categorical_segmentation"},
    # 2 autoencoders
    "identity_grey": {"type": "identity"},
    "identity_color": {"type": "identity"},
    # 8 rectangles
    "crystal_bbox": {"type": "rectangle_regression"},
    "loop_inside_bbox": {"type": "rectangle_regression"},
    "loop_bbox": {"type": "rectangle_regression"},
    "stem_bbox": {"type": "rectangle_regression"},
    "pin_bbox": {"type": "rectangle_regression"},
    "foreground_bbox": {"type": "rectangle_regression"},
    "area_of_interest_bbox": {"type": "rectangle_regression"},
    "support_bbox": {"type": "rectangle_regression"},
    # 8 ellipses
    "crystal_ellipse": {"type": "ellipse_regression"},
    "loop_inside_ellipse": {"type": "ellipse_regression"},
    "loop_ellipse": {"type": "ellipse_regression"},
    "stem_ellipse": {"type": "ellipse_regression"},
    "pin_ellipse": {"type": "ellipse_regression"},
    "foreground_ellipse": {"type": "ellipse_regression"},
    "area_of_interest_ellipse": {"type": "ellipse_regression"},
    "support_ellipse": {"type": "ellipse_regression"},
    # 4
    "most_likely_point": {"type": "point_regression"},
    "extreme_point": {"type": "point_regression"},
    "start_likely_point": {"type": "point_regression"},
    "start_possible_point": {"type": "point_regression"},
}

notion_importance = {
    "user_click": 1,
    "crystal": 1,
    "loop_inside": 2,
    "loop": 3,
    "area_of_interest": 4,
    "stem": 5,
    "support": 6,
    "support_filled": 7,
    "pin": 8,
    "capillary": 9,
    "ice": 10,
    "dust": 11,
    "drop": 12,
    "foreground": 13,
    "not_background": 14,
    "background": 100.0,
}

line_styles = {
    "loop_inside": "dotted",
    "loop": "dotted",
    "area_of_interest": "dashdot",
    "stem": "dotted",
    "support": "dashed",
    "support_filled": "dashdot",
    "ice": "dotted",
    "crystal": "dotted",
    "pin": "dashed",
}

# {
# "crystal": 1,
# "loop_inside": 2,
# "loop": 3,
# "stem": 4,
# "pin": 5,
# "capillary": 6,
# "ice": 7,
# "dust": 8,
# "drop": 9,
# "foreground": 10,
# "not_background": 11,
# "background": 100.0,
# },


def timeit(f):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    def timed(*args, **kw):
        ts = time.time()
        result = f(*args, **kw)
        te = time.time()

        print("func:%r took: %2.8f sec" % (f.__name__, te - ts))
        return result

    return timed


def get_image(json_file):
    imageData = json_file.get("imageData")
    if imageData is not None:
        image = utils.img_b64_to_arr(imageData) / 255.0
    else:
        image = imageio.imread(json_file.get("imagePath"))
    return image


def get_shapes(json_file):
    shapes = json_file.get("shapes")
    return shapes


def get_image_shape(json_file):
    image_shape = np.array((json_file.get("imageHeight"), json_file.get("imageWidth")))
    return image_shape


# @timeit
def get_hierarchical_mask(
    json_file,
    notions=[
        "crystal",
        # "ice",
        # "dust"
        "loop_inside",
        "loop",
        "pin",
        "stem",
        # "capillary",
        # "drop",
        "foreground",
        "background",
    ],
    notion_importance=notion_importance,
):
    notions.sort(key=lambda x: -notion_importance[x])
    notion_values = np.array([notion_importance[notion] for notion in notions])

    oois = get_objects_of_interest(json_file)
    image_shape = oois["image_shape"]
    hierarchical_target = np.zeros(tuple(image_shape) + (len(notions),))

    for label in oois:
        if label in ["image_shape", "user_click", "background", "labeled_points"]:
            continue
        label_mask = get_label_mask(oois, [label])
        if label in notions:
            i = notions.index(label)
        elif label != "background":
            i = notions.index("foreground")
        hierarchical_target[:, :, i] = np.logical_or(
            hierarchical_target[:, :, i], label_mask
        )

    hierarchical_target /= notion_values
    hierarchical_mask = np.argmax(hierarchical_target, axis=2)
    return hierarchical_mask


# @timeit
def get_hierarchy_from_oois(
    oois,
    points=None,
    notions=[
        "crystal",
        # "ice",
        # "dust"
        "loop_inside",
        "loop",
        "pin",
        "stem",
        # "capillary",
        # "drop",
        "foreground",
        "background",
    ],
    notion_importance={
        "crystal": 1,
        "loop_inside": 2,
        "loop": 3,
        "stem": 4,
        "pin": 5,
        "capillary": 6,
        "ice": 7,
        "dust": 8,
        "drop": 9,
        "foreground": 10,
        "not_background": 11,
        "background": 100.0,
    },
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

    hierarchical_target /= notion_values
    hierarchical_mask = np.argmax(hierarchical_target, axis=2)
    return hierarchical_mask


##@timeit
def get_hierarchy_from_masks(
    masks,
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "pin",
        "stem",
        "foreground",
        "background",
    ],
    notion_importance={
        "crystal": 1,
        "loop_inside": 2,
        "loop": 3,
        "stem": 4,
        "pin": 5,
        "capillary": 6,
        "ice": 7,
        "dust": 8,
        "drop": 9,
        "foreground": 10,
        "not_background": 11,
        "background": 100.0,
    },
):
    notions.sort(key=lambda x: -notion_importance[x])
    notion_values = np.array([notion_importance[notion] for notion in notions])

    hierarchical_target = np.zeros(masks["foreground"].shape + (len(notions),))

    for notion in notions:
        i = notions.index(notion)
        if notion in masks:
            hierarchical_target[:, :, i] = masks[notion]
        elif notion == "background":
            hierarchical_target[:, :, i] = 1.0

    hierarchical_target /= notion_values
    hierarchical_mask = np.argmax(hierarchical_target, axis=2)
    return hierarchical_mask


# @timeit
def get_label_mask(oois, labels):
    image_shape = oois["image_shape"]
    print(f"image_shape {image_shape} {type(image_shape)}")
    label_mask = np.zeros(image_shape, dtype=np.uint8)

    for label in labels:
        print(f"label {label}")
        if label not in oois or label in ["image", "labels"]:
            continue
        for points in oois[label]:
            if len(points) < 3:
                continue
            print(f"points {points}")
            polygon = points * image_shape
            mask = get_mask_from_polygon(polygon, image_shape)
            label_mask = np.logical_or(label_mask == 1, mask == 1)
    return label_mask


# @timeit
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


def get_label_polygons(oois, labels, points=None):
    polygons = []
    label_list = oois["labels"]
    label_indices = oois["indices"]
    image_shape = oois["image_shape"]
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
            polygons.append(polygon)
    return polygons


# @timeit
def get_mask_rectangles_ellipses_from_points(oois, labels, points=None):
    image_shape = oois["image_shape"]
    label_mask = np.zeros(image_shape, dtype=np.uint8)
    rectangles = []
    ellipses = []

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
            rps = get_rps(mask)
            rectangle = get_rectangle_from_rps(rps)
            ellipse = get_ellipse_from_rps(rps)

            label_mask = np.logical_or(label_mask == 1, mask == 1)
            rectangles.append(rectangle)
            ellipses.append(ellipse)

    if len(rectangles) > 1:
        rps = get_rps(label_mask)
        rectangle = get_rectangle_from_rps(rps)
        ellipse = get_ellipse_from_rps(rps)
        rectangles.append(rectangle)
        ellipses.append(ellipse)

    return label_mask, rectangles, ellipses


# @timeit
def get_targets_old(
    oois,
    points=None,
    notions={
        # 8 binary segmentations
        "crystal": {"type": "binary_segmentation"},
        "loop_inside": {"type": "binary_segmentation"},
        "loop": {"type": "binary_segmentation"},
        "stem": {"type": "binary_segmentation"},
        "pin": {"type": "binary_segmentation"},
        "foreground": {"type": "binary_segmentation"},
        "ice": {"type": "binary_segmentation"},
        "area_of_interest": {"type": "binary_segmentation"},
        "support": {"type": "binary_segmentation"},
        # 1 categorical segmentation
        "hierarchy": {"type": "categorical_segmentation"},
        # 2 autoencoders
        "identity_grey": {"type": "identity"},
        "identity_color": {"type": "identity"},
        # 8 rectangles
        "crystal_bbox": {"type": "rectangle_regression"},
        "loop_inside_bbox": {"type": "rectangle_regression"},
        "loop_bbox": {"type": "rectangle_regression"},
        "stem_bbox": {"type": "rectangle_regression"},
        "pin_bbox": {"type": "rectangle_regression"},
        "foreground_bbox": {"type": "rectangle_regression"},
        "area_of_interest_bbox": {"type": "rectangle_regression"},
        "support_bbox": {"type": "rectangle_regression"},
        # 8 ellipses
        "crystal_ellipse": {"type": "ellipse_regression"},
        "loop_inside_ellipse": {"type": "ellipse_regression"},
        "loop_ellipse": {"type": "ellipse_regression"},
        "stem_ellipse": {"type": "ellipse_regression"},
        "pin_ellipse": {"type": "ellipse_regression"},
        "foreground_ellipse": {"type": "ellipse_regression"},
        "area_of_interest_ellipse": {"type": "ellipse_regression"},
        "support_ellipse": {"type": "ellipse_regression"},
        # 4
        "most_likely_point": {"type": "point_regression"},
        "extreme_point": {"type": "point_regression"},
        "start_likely_point": {"type": "point_regression"},
        "start_possible_point": {"type": "point_regression"},
    },
):
    targets = {}
    for notion in notions:
        if notions[notion]["type"] == "binary_segmentation":
            bbox = "%s_bbox" % notion
            ellipse = "%s_ellipse" % notion
            rectangles = None
            ellipses = None
            if notion == "crystal":
                (
                    label_mask,
                    rectangles,
                    ellipses,
                ) = get_mask_rectangles_ellipses_from_points(
                    oois, [notion], points=points
                )
                targets[notion] = label_mask
                targets[bbox] = rectangles
                targets[ellipse] = ellipses
            elif notion == "area_of_interest":
                targets[notion] = get_aoi(oois, points=points)
            elif notion == "support":
                targets[notion] = get_support(oois, points=points)
            elif notion == "foreground":
                targets[notion] = get_foreground(oois, points=points)
            else:
                targets[notion] = get_label_mask_from_points(
                    oois, [notion], points=points
                )

            if (
                bbox in notions
                or ellipse in notions
                and (rectangles is None or ellipses is None)
            ):
                polygons = get_label_polygons(oois, [notion], points=points)
                print("notion %s polygons" % notion, polygons)
                if polygons:
                    rps = cvRegionprops(polygons[0])
                # rps = get_rps(targets[notion])
                if bbox in notions:
                    if type(rps) is cvRegionprops:
                        rectangles = rps.get_bbox()
                    else:
                        rectangles = [get_rectangle_from_rps(rps)]
                    targets[bbox] = rectangles
                if ellipse in notions:
                    if type(rps) is cvRegionprops:
                        ellipses = rps.get_ellipse()
                    else:
                        ellipses = [get_ellipse_from_rps(rps)]
                    targets[ellipse] = ellipses
        if points is None:
            points = oois["points"]

        elif notions[notion]["type"] == "point_regression":
            point_index = oois["labels"].index(notion)
            i_start, i_end = oois["indices"][point_index]
            targets[notion] = points[i_start:i_end]

    targets["hierarchy"] = get_hierarchy_from_masks(targets)
    targets["identity_color"] = oois["image"]
    targets["identity_grey"] = oois["image"].mean(axis=2)

    return targets


# from scratch import get_targets


def get_targets(
    oois,
    points=None,
    notions={
        "primary": [
            "crystal",
            "loop_inside",
            "loop",
            "stem",
            "pin",
            # "ice",
            # "dust",
            "area_of_interest",
            "support",
            "foreground",
        ],
        "secondary": ["hierarchy", "identity", "identity_grey"],
        "tertiary": [
            "most_likely_point",
            "extreme_point",
            "start_likely_point",
            "start_possible_point",
        ],
    },
    notion_importance={
        "crystal": 1,
        "loop_inside": 2,
        "loop": 3,
        "stem": 4,
        "pin": 5,
        "capillary": 6,
        "ice": 7,
        "dust": 8,
        "drop": 9,
        "foreground": 10,
        "not_background": 11,
        "background": 100.0,
    },
    notion_hierarchy_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
        "background": 6,
    },
    debug=False,
):
    targets = {}
    image_shape = oois["image_shape"]
    if points is None:
        points = oois["points"]

    mask_template = np.zeros(image_shape, dtype=np.uint8)
    masks = {}
    for notion in notions["primary"]:
        masks[notion] = np.zeros(image_shape, dtype=np.uint8)

    props = []
    for k, label in enumerate(oois["labels"]):
        i_start, i_end = oois["indices"][k]
        ps = points[i_start:i_end]
        if len(ps) < 3:
            continue

        mask = get_mask_from_polygon(ps, image_shape=image_shape)

        if label in notions["primary"]:
            prop = get_regionprops(polygon)
            prop["label"] = label
            props.append(prop)
            masks[label] = np.logical_or(masks[label], mask) if label in masks else mask

    if "loop_inside" in masks:
        masks["support"] = np.logical_xor(masks["support"], masks["loop_inside"])

    notions_list = list(notion_hierarchy_indices.keys())
    notions_list.sort(key=lambda x: -notion_hierarchy_indices[x])

    hierarchy = np.zeros(image_shape, dtype=np.uint8)
    for notion in notions_list:
        if notion in masks:
            hierarchy[masks[notion] == 1] = notion_hierarchy_indices[notion]
        elif notion == "background":
            hierarchy[masks["foreground"] != 1] = notion_hierarchy_indices["background"]

    masks["hierarchy"] = hierarchy
    image = oois["image"]
    masks["identity"] = image
    masks["identity_grey"] = image.mean(axis=2)
    targets["masks"] = masks
    targets["props"] = props

    if debug:
        fig, axes = pylab.subplots(1, 4)
        axes[0].imshow(hierarchy)
        axes[1].imshow(masks["support"])
        axes[2].imshow(masks["area_of_interest"])
        axes[3].imshow(masks["foreground"])
        pylab.show()

    return targets


# https://stackoverflow.com/questions/74759071/draw-area-polygon-with-a-hole-in-opencv
def get_complex_mask(
    oois, positive=["crystal", "loop_inside", "loop"], negative=[], points=None
):
    positive_mask = get_label_mask_from_points(oois, positive, points=points)
    negative_mask = get_label_mask_from_points(oois, negative, points=points)
    complex_mask = np.logical_xor(positive_mask, negative_mask)
    return complex_mask


def get_aoi(oois, points=None):
    aoi = get_label_mask_from_points(
        oois, ["crystal", "loop_inside", "loop", "cd_loop"], points=points
    )
    return aoi


get_area_of_interest_mask = get_aoi


def get_foreground(oois, points=None):
    foreground = get_label_mask_from_points(oois, ["any"], points=points)
    return foreground


def get_support(oois, points=None):
    positive = get_label_mask_from_points(
        oois, ["stem", "loop", "cd_loop", "cd_stem"], points=points
    )
    negative = get_label_mask_from_points(oois, ["loop_inside"], points=points)
    support = np.logical_xor(positive, negative)
    return support


get_support_mask = get_support


# @timeit
def get_extreme_point(
    projection, pa=None, orientation="horizontal", extreme_direction=1
):
    try:
        xyz = np.argwhere(projection != 0)
        if pa is None:
            pa = principal_axes(projection)

        S = pa[-2]
        center = pa[-1]

        xyz_0 = xyz - center

        xyz_S = np.dot(xyz_0, S)
        xyz_S_on_axis = xyz_S[np.isclose(xyz_S[:, 1], 0, atol=5)]

        mino = xyz_S[np.argmin(xyz_S[:, 0])]
        try:
            mino_on_axis = xyz_S_on_axis[np.argmin(xyz_S_on_axis[:, 0])]
        except BaseException:
            print(traceback.print_exc())
            mino_on_axis = copy.copy(mino)
        maxo = xyz_S[np.argmax(xyz_S[:, 0])]
        try:
            maxo_on_axis = xyz_S_on_axis[np.argmax(xyz_S_on_axis[:, 0])]
        except BaseException:
            print(traceback.print_exc())
            maxo_on_axis = copy.copy(maxo)

        mino_0_s = np.dot(mino, np.linalg.inv(S)) + center
        maxo_0_s = np.dot(maxo, np.linalg.inv(S)) + center

        mino_0_s_on_axis = np.dot(mino_on_axis, np.linalg.inv(S)) + center
        maxo_0_s_on_axis = np.dot(maxo_on_axis, np.linalg.inv(S)) + center

        if orientation == "horizontal":
            if extreme_direction * mino_0_s[1] > extreme_direction * maxo_0_s[1]:
                extreme_point_out = mino_0_s
                extreme_point_out_on_axis = mino_0_s_on_axis
                extreme_point_ini = maxo_0_s
                extreme_point_ini_on_axis = maxo_0_s_on_axis
            else:
                extreme_point_out = maxo_0_s
                extreme_point_out_on_axis = maxo_0_s_on_axis
                extreme_point_ini = mino_0_s
                extreme_point_ini_on_axis = mino_0_s_on_axis
        else:
            if extreme_direction * mino_0_s[0] > extreme_direction * maxo_0_s[0]:
                extreme_point_out = mino_0_s
                extreme_point_out_on_axis = mino_0_s_on_axis
                extreme_point_ini = maxo_0_s
                extreme_point_ini_on_axis = maxo_0_s_on_axis
            else:
                extreme_point_out = maxo_0_s
                extreme_point_out_on_axis = maxo_0_s_on_axis
                extreme_point_ini = mino_0_s
                extreme_point_ini_on_axis = mino_0_s_on_axis
    except BaseException:
        print(traceback.print_exc())
        (
            extreme_point_out,
            extreme_point_ini,
            extreme_point_out_on_axis,
            extreme_point_ini_on_axis,
        ) = [[-1, -1]] * 4
    return (
        extreme_point_out,
        extreme_point_ini,
        extreme_point_out_on_axis,
        extreme_point_ini_on_axis,
        pa,
    )


# @timeit
def get_ellipse_from_polygon(polygon):
    ## https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
    # Y = polygon[:, 0:1]
    # X = polygon[:, 1:]
    # A = np.hstack([X ** 2, X * Y, Y ** 2, X, Y])
    # print("A.shape", A.shape)
    # print("A")
    # print(A)
    # b = np.ones_like(X)
    # x = np.linalg.lstsq(A, b)
    # print("solution ", x)
    # solution = x[0].squeeze()
    ellipse = cv.fitEllipse(polygon.astype(int))
    print(f"ellipse {ellipse}")
    return ellipse


# @timeit
def get_rps(mask):
    rps = ski.measure.regionprops(ski.measure.label(mask))[0]
    return rps


# https://docs.opencv.org/4.x/d1/d32/tutorial_py_contour_properties.html
class cvRegionprops(object):
    def __init__(self, points=None):
        self.points = points[::-1].astype(np.int32)
        self.bbox = None
        self.centroid = None
        self.min_rectangle = None
        self.min_enclosing_circle = None
        self.ellipse = None
        self.area = None
        self.perimeter = None
        self.moments = None

    def get_centroid(self):
        return self.points.mean(axis=0)

    def get_bbox(self):
        # self.bbox = get_rectangle_from_polygon(self.points) #
        self.bbox = cv.boundingRect(self.points)
        return self.bbox

    def get_ellipse(self):
        self.ellipse = cv.fitEllipse(self.points)
        return self.ellipse

    def get_min_rectangle(self):
        self.min_rectangle = cv.minAreaRect(self.points)
        return self.min_rectangle

    def get_min_enclosing_circle(self):
        self.min_enclosing_circle = cv.minEnclosingCircle(self.points)
        return self.min_enclosing_circle

    def get_area(self):
        self.area = cv.contourArea(self.points)
        return self.area

    def get_perimeter(self):
        self.perimeter = cv.arcLength(self.points, True)
        return self.perimeter

    def get_moments(self):
        self.moments = cv.moments(self.points)
        return self.moments

    def get_extreme_points(self):
        self.extreme_points = get_extreme_points(self.points)
        return self.extreme_points

    def get_extreme_points_eigen(self):
        center = np.mean(self.points, axis=0)
        coord = self.points - center
        inertia = np.dot(coord.transpose(), coord)
        e_values, e_vectors = np.linalg.eig(inertia)
        order = np.argsort(e_values)[::-1]
        # eigenvalues = np.array(e_values[order])
        S = np.array(e_vectors[:, order])
        coord_S = np.dot(coord, S)
        extreme_points_S = get_extreme_points(coord_S)
        extreme_points_O = np.dot(extreme_points, np.linalg.inv(S)) + center
        self.extreme_points_eigen = get_extreme_points(extreme_points_O)
        return self.extreme_points_eigen

    def get_aspect(self):
        x, y, w, h = cv.boundingRect(cnt)
        self.aspect = float(w) / h

    def get_extent(self):
        area = cv.contourArea(cnt)
        x, y, w, h = cv.boundingRect(cnt)
        rect_area = w * h
        self.extent = float(area) / rect_area
        return self.extent

    def get_solidity(self):
        area = cv.contourArea(cnt)
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        self.solidity = float(area) / hull_area


# @timeit
def principal_axes(array, verbose=False):
    # https://github.com/pierrepo/principal_axes/blob/master/principal_axes.py
    _start = time.time()
    if array.shape[1] != 3:
        xyz = np.argwhere(array == 1)
    else:
        xyz = array[:, :]

    coord = np.array(xyz, float)
    center = np.mean(coord, 0)
    coord = coord - center
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)
    order = np.argsort(e_values)[::-1]
    eigenvalues = np.array(e_values[order])
    eigenvectors = np.array(e_vectors[:, order])
    _end = time.time()
    if verbose:
        print("principal axes")
        print("intertia tensor")
        print(inertia)
        print("eigenvalues")
        print(eigenvalues)
        print("eigenvectors")
        print(eigenvectors)
        print("principal_axes calculated in %.4f seconds" % (_end - _start))
        print()
    return inertia, eigenvalues, eigenvectors, center


# @timeit
def get_ellipse_from_mask(mask):
    rps = get_rps(mask)
    r, c = rps.centroid
    major = rps.axis_major_length
    minor = rps.axis_minor_length
    orientation = rps.orientation
    return r, c, major, minor, orientation


# @timeit
def get_ellipse_from_rps(rps):
    r, c = rps.centroid
    major = rps.axis_major_length
    minor = rps.axis_minor_length
    orientation = rps.orientation
    return r, c, major, minor, orientation


# @timeit
def get_mask_from_polygon(polygon, image_shape=(1200, 1600), doer="cv"):
    if doer == "ski":
        mask = polygon2mask(image_shape, polygon)
    else:
        mask = cv.fillPoly(
            np.zeros(image_shape, np.uint8), [polygon[:, ::-1].astype(np.int32)], 1
        )
    return mask


def get_labelme_shape_from_mask(mask, label):
    shape = {}
    shape["label"] = label
    shape["group_id"] = None
    shape["description"] = ""
    shape["flags"] = {}
    shape["mask"] = None
    contours = cv.findContours(
        mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    points = contours[0][0].astype(float)
    points = points.reshape((points.shape[0], points.shape[-1]))
    # points = points[:, ::-1]
    pts = []
    for p in points:
        pts.append(list(p))
    if len(points) >= 3:
        shape_type = "polygon"
    elif len(points) == 2:
        shape_type = "rectangle"
    else:
        shape_type = "point"
    shape["shape_type"] = shape_type
    shape["points"] = pts
    return shape


def get_new_json_file(
    shapes,  # list of shape directories
    imagePath,
    imageHeight=None,
    imageWidth=None,
    version="5.4.1",
    imageData=None,
    flags={},
):
    json_file = {}
    json_file["version"] = version
    json_file["flags"] = flags
    json_file["shapes"] = shapes
    json_file["imagePath"] = imagePath
    if imageHeight is None or imageWidth is None:
        img = imageio.imread(imagePath)
        imageHeight, imageWidth = img.shape[:2]
    json_file["imageHeight"] = imageHeight
    json_file["imageWidth"] = imageWidth
    json_file["imageData"] = imageData

    return json_file


def get_labelme_shapes_from_chimp_record(imagepath):
    realpath = os.path.realpath(imagepath)
    record = np.load(realpath.replace("images", "masks").replace(".jpg", ".npz"))
    labels = record["class_labels"]
    masks = record["masks"]
    shapes = []
    for mask, label in zip(masks, labels):
        shape = get_labelme_shape_from_mask(mask, label)
        shapes.append(shape)

    return shapes


def create_labelme_file_from_chimp_record(imagepath):
    realpath = os.path.realpath(imagepath)
    shapes = get_labelme_shapes_from_chimp_record(realpath)
    jsonpath = realpath.replace("images", "json").replace(".jpg", ".json")

    json_file = get_new_json_file(shapes, realpath)

    if not os.path.isdir(os.path.dirname(jsonpath)):
        os.makedirs(os.path.dirname(jsonpath))
    fp = open(jsonpath, "w")
    json.dump(json_file, fp)
    fp.close()


# this may be superfluous ? labels, indices, points, already contain everything, may save a little time when generating examples on the fly but it is probably negligible
# if label not in objects_of_interest:
# objects_of_interest[label] = []
# objects_of_interest[label].append(ooi)


def add_ooi(ooi, label, points, indices, labels):
    if indices:
        i_start = indices[-1][-1]
    else:
        i_start = 0
    i_end = i_start + ooi.shape[0]
    points = np.vstack([points, ooi]) if len(points) else ooi
    indices.append((i_start, i_end))
    labels.append(label)

    return points, indices, labels


# @timeit
def get_objects_of_interest(json_file, fractional=False):
    image = get_image(json_file)
    image_shape = np.array(image.shape[:2])

    points, indices, labels = [], [], []

    for shape in get_shapes(json_file):
        label = shape["label"]
        if label in additional_labels:
            label = additional_labels[label]
        ooi = np.array(shape["points"])
        ooi = ooi[
            :, ::-1  # swap x and y (labelme uses [h, v] convention, we use [v, h]
        ]
        if fractional:
            ooi /= image_shape
        points, indices, labels = add_ooi(ooi, label, points, indices, labels)

    if "background" not in labels:
        background = [[0, 0], [0, 1], [1, 1], [0, 1]]
        background = np.array(background)
        if not fractional:
            ooi = background * image_shape
        points, indices, labels = add_ooi(ooi, "background", points, indices, labels)

    points, indices, labels = get_secondary_notions(
        points, indices, labels, image_shape, fractional=fractional
    )

    objects_of_interest = {
        "image": image,
        "image_shape": image_shape,
        "fractional": fractional,
        "labels": labels,
        "indices": indices,
        "points": points,
    }
    return objects_of_interest


def update_masks(masks, label, mask):
    masks[label] = (
        np.logical_or(masks[label], mask) if label in masks else mask
    ).astype(np.uint8)
    return masks


def get_masks(points, indices, labels, image_shape, fractional=False, image=None):
    masks = {}

    for k, label in enumerate(labels):
        if label == "background":
            continue
        i_start, i_end = indices[k]
        ps = points[i_start:i_end]
        if len(ps) < 3:
            continue
        if fractional:
            ps *= image_shape

        mask = get_mask_from_polygon(ps, image_shape=image_shape)

        masks = update_masks(masks, label, mask)
        masks = update_masks(masks, "foreground", mask)

        if label in ["crystal", "loop"]:
            masks = update_masks(masks, "area_of_interest", mask)

        if label in ["loop", "stem"]:
            masks = update_masks(masks, "support", mask)

    if "support" in masks and "pin" in masks:
        masks["support"][masks["pin"].astype(bool)] = 0

    return masks


def get_secondary_notions(
    points,
    indices,
    labels,
    image_shape,
    fractional=False,
    secondary_notions=["area_of_interest", "support", "foreground"],
):
    masks = get_masks(points, indices, labels, image_shape)

    for notion in secondary_notions:
        if notion in masks:
            contours, h = cv.findContours(
                masks[notion].astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
            )
            shape = contours[0].shape
            new_shape = (shape[0], shape[2])
            ooi = contours[0].reshape(new_shape)[:, ::-1]
            if fractional:
                ooi /= image_shape
            points, indices, labels = add_ooi(ooi, notion, points, indices, labels)

    return points, indices, labels


def add_to_keypoint_labels_and_points(
    notion,
    polygon,
    image_shape,
    keypoint_labels,
    keypoint_points,
    keypoints=["left", "right", "top", "bottom"],
):
    extreme_points_eigen = np.reshape(get_extreme_points_eigen(polygon), (4, 2))
    for name, point in zip(keypoints, extreme_points_eigen):
        label = f"{notion}_{name}_point"
        keypoint_labels.append(label)
        keypoint_points.append(point[::-1] / image_shape)
    centroid_point = np.mean(polygon[:, ::-1] / image_shape, axis=0)
    centroid_label = f"{notion}_centroid_point"
    keypoint_labels.append(centroid_label)
    keypoint_points.append(centroid_point)

    return keypoint_labels, keypoint_points


def add_critical_keypoints(labels, indices, points):
    most_likely_point = get_most_likely_point(labels, indices, points)
    extreme_point = get_point("foreground_right_point", labels, indices, points)
    start_likely_point = get_point(
        "area_of_interest_left_point", labels, indices, points
    )
    start_possible_point = get_start_possible_point(labels, indices, points)

    keypoint_labels = [
        "most_likely_point",
        "extreme_point",
        "start_likely_point",
        "start_possible_point",
    ]

    keypoint_points = [
        most_likely_point,
        extreme_point,
        start_likely_point,
        start_possible_point,
    ]

    labels, indices, points = add_keypoints(
        keypoint_labels, keypoint_points, labels, indices, points
    )

    return labels, indices, points


def add_keypoints(
    keypoint_labels, keypoint_points, labels, indices, points, debug=False
):
    i_start, i_end = indices[-1]
    for label, point in zip(keypoint_labels, keypoint_points):
        i_start = i_end
        if debug:
            print(label, point)
        i_end = i_start + 1
        point_index = (i_start, i_end)
        labels.append(label)
        indices.append(point_index)
        points = np.vstack([points, point])

    return labels, indices, points


def get_point(label, labels, indices, points):
    point = np.array([-1, -1])
    if label in labels:
        k = labels.index(label)
        i_start, i_end = indices[k]
        point = points[i_start:i_end]

    return point.squeeze()


def get_start_possible_point(labels, indices, points):
    start_possible_point = get_point("support_left_point", labels, indices, points)
    end_pin = get_point("pin_right_point", labels, indices, points)
    extreme = get_point("foreground_right_point", labels, indices, points)
    if start_possible_point[0] != -1 and end_pin[0] != -1 and extreme[0] != -1:
        o = np.array(extreme)
        pin = np.linalg.norm(np.array(end_pin) - o)
        sup = np.linalg.norm(np.array(start_possible_point) - o)
        if pin < sup:
            start_possible_point = end_pin
    else:
        start_possible_point = get_point(
            "foreground_left_point", labels, indices, points
        )

    return start_possible_point


# @timeit
def get_most_likely_point(labels, indices, points):
    for label in [
        "largest_crystal_centroid_point",
        "user_click",
        "area_of_interest_centroid_point",
        "foreground_right_point",
    ]:
        most_likely_point = get_point(label, labels, indices, points)
        if most_likely_point[0] != -1 and most_likely_point[1] != -1:
            break

    return most_likely_point


# @timeit
def get_keypoints(oois):
    labels = [
        "most_likely_point",
        "extreme_point",
        "start_likely_point",
        "start_possible_point",
        "top_left",
        "top_right",
        "bottom_right",
        "bottom_left",
    ]
    most_likely_point = get_most_likely_point(oois)

    aoi = get_aoi_mask(oois)
    epo, epi, epooa, epioa, pa = get_extreme_point(aoi)
    extreme_point = epooa / oois["image_shape"]
    start_likely_point = epioa / oois["image_shape"]

    support = get_support_mask(oois)
    epo, epi, epooa, epioa, pa = get_extreme_point(support)
    start_possible_point = epioa / oois["image_shape"]
    keypoints = [
        most_likely_point,
        extreme_point,
        start_likely_point,
        start_possible_point,
        np.array([0, 0]),
        np.array([0, 1]),
        np.array([1, 1]),
        np.array([1, 0]),
    ]
    return labels, keypoints


def get_regionprops(points):
    regionprops = {}
    cx, cy = np.mean(points, axis=0)
    x, y, w, h = cv.boundingRect(points)
    (mx, my), (mw, mh), mo = cv.minAreaRect(points)
    (ex, ey), (eMA, ema), eo = cv.fitEllipse(points)
    (lx, ly), (rx, ry), (tx, ty), (bx, by) = get_extreme_points(points)
    (elx, ely), (erx, ery), (etx, ety), (ebx, eby) = get_extreme_points_eigen(points)
    area = cv.contourArea(points)
    perimeter = cv.arcLength(points, True)
    (ecx, ecy), ecr = cv.minEnclosingCircle(points)
    moments = cv.moments(points)
    aspect = float(w) / h
    rect_area = w * h
    extent = float(area) / rect_area
    hull = cv.convexHull(points)
    hull_area = cv.contourArea(hull)
    solidity = -1
    if hull_area:
        solidity = float(area) / hull_area

    regionprops["centroid"] = cx, cy
    regionprops["bbox"] = x, y, w, h
    regionprops["min_rectangle"] = mx, my, mw, mh, mo
    regionprops["ellipse"] = ex, ey, eMA, ema, eo
    regionprops["extreme_points"] = lx, ly, rx, ry, tx, ty, bx, by
    regionprops["extreme_points_eigen"] = elx, ely, erx, ery, etx, ety, ebx, eby
    regionprops["area"] = area
    regionprops["perimeter"] = perimeter
    regionprops["min_enclosing_circle"] = ecx, ecy, ecr
    regionprops["moments"] = moments
    regionprops["aspect"] = aspect
    regionprops["rect_area"] = rect_area
    regionprops["extent"] = extent
    regionprops["solidity"] = solidity

    return regionprops


def get_extreme_points(cnt):
    leftmost = cnt[cnt[:, 0] == cnt[cnt[:, 0].argmin()][0]].mean(axis=0)
    rightmost = cnt[cnt[:, 0] == cnt[cnt[:, 0].argmax()][0]].mean(axis=0)
    topmost = cnt[cnt[:, 1] == cnt[cnt[:, 1].argmin()][1]].mean(axis=0)
    bottommost = cnt[cnt[:, 1] == cnt[cnt[:, 1].argmax()][1]].mean(axis=0)
    return leftmost, rightmost, topmost, bottommost


def get_extreme_points_eigen(points):
    center = np.mean(points, axis=0)
    coord = points - center
    inertia = np.dot(coord.transpose(), coord)
    e_values, e_vectors = np.linalg.eig(inertia)
    order = np.argsort(e_values)[::-1]
    S = np.array(e_vectors[:, order])
    coord_S = np.dot(coord, S)
    extreme_points_S = get_extreme_points(coord_S)
    extreme_points_O = np.dot(extreme_points_S, np.linalg.inv(S)) + center
    extreme_points_eigen = get_extreme_points(extreme_points_O)
    return extreme_points_eigen


# @timeit
def get_rectangle(bbox, encoding="preferred"):
    if encoding in ["preferred", "matplotlib"]:
        pvmin, phmin, pvmax, phmax = bbox
        extent_v = pvmax - pvmin
        extent_h = phmax - phmin
    if encoding == "preferred":
        center_v = pvmin + extent_v / 2
        center_h = phmin + extent_h / 2
        rectangle = [center_v, center_h, extent_v, extent_h]
    elif encoding == "matplotlib":
        rectangle = [phmin, pvmin, extent_h, extent_v]
    else:
        rectangle = [pvmin, phmin, pvmax, phmax]
    return rectangle


# @timeit
def get_rectangle_from_rps(rps, encoding="preferred"):
    rectangle = get_rectangle(rps.bbox, encoding=encoding)
    return rectangle


# @timeit
def get_rectangle_from_polygon(polygon, encoding="preferred"):
    pvmax = polygon[:, 0].max()
    pvmin = polygon[:, 0].min()
    phmax = polygon[:, 1].max()
    phmin = polygon[:, 1].min()
    bbox = [pvmin, phmin, pvmax, phmax]
    rectangle = get_rectangle(bbox, encoding=encoding)
    return rectangle


# @timeit
def get_support_mask(oois):
    support = get_label_mask(oois, ["loop", "stem", "cd_loop", "cd_stem"]).astype(int)
    nsupport = get_label_mask(oois, ["loop_inside"])
    support_mask = np.logical_xor(support, nsupport)
    return support_mask


# @timeit
def get_aoi_mask(oois):
    aoi_mask = get_label_mask(
        oois, ["crystal", "loop_inside", "loop", "cd_loop", "cd_stem"]
    )
    return aoi_mask


# @timeit
def make_points_homogeneous(points):
    hpoints = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    return hpoints


def get_corners():
    corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1], [0.5, 0.5]])
    return corners


# @timeit
def get_output_shape(input_shape, transformation_matrix):
    corners = get_corners()
    print("corners")
    print(corners)
    print(f"input_shape {input_shape}")
    print(f"transformation_matrix")
    print(transformation_matrix)
    corners *= input_shape
    print(f"corners {corners}")
    hcorners = make_points_homogeneous(corners)
    print(f"hcorners {hcorners}")
    tcorners = get_transformed_points(hcorners, transformation_matrix)
    print(f"tcorners {tcorners}")
    # distances = np.abs(tcorners[:-1, :2] - tcorners[-1, :2])
    # print(f'distances {distances}')
    output_shape = np.max(tcorners[:, :2], axis=0)
    print(f"output_shape {output_shape}")
    return output_shape.astype(int)


# @timeit
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


# def plot_keypoints(ax, keypoints):
# colors = colors_for_labels.keys()
# for k, point in enumerate(keypoints):
# patch = pylab.Circle(point[:2], radius=7, color=sns.xkcd_rgb[colors[k]])
# ax.add_patch(patch)


def plot_keypoints(keypoints, radius=1, colors=xkcd_colors_that_i_like, ax=None):
    if ax is None:
        ax = pylab.gca()
    for k, p in enumerate(keypoints):
        c = pylab.Circle(p[:2][::-1], radius=radius, color=sns.xkcd_rgb[colors[k]])
        ax.add_patch(c)


# @timeit
def get_transformed_points(points, transformation_matrix):
    points = points[:, [1, 0, 2]]
    transformed_points = np.dot(transformation_matrix, points.T).T
    transformed_points = transformed_points[:, [1, 0, 2]]
    return transformed_points


# @timeit
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


def plot_oois(points, labels, radius=7, ax=None):
    if ax is None:
        ax = pylab.gca()
    print(f"points {len(points)}")
    print(f"labels {labels}")
    for label, i_start, i_end in labels:
        # if label not in ['crystal']: #'user_click', 'pin', 'stem']:
        # continue
        print(f"label {label}, i_start {i_start}, i_end {i_end}")
        color = sns.xkcd_rgb[colors_for_labels[label]]
        ps = points[i_start:i_end, :]
        matlab_ps = ps[:, ::-1]
        if len(ps) >= 3:
            patch = pylab.Polygon(matlab_ps, color=color, lw=2, fill=False)
            ax.add_patch(patch)
            x, y, width, height = get_rectangle_from_polygon(ps, encoding="matplotlib")
            patch = pylab.Rectangle(
                (x, y), width, height, color=color, lw=2, fill=False
            )
            ax.add_patch(patch)
        elif len(ps) == 1:
            print("point!")
            print(matlab_ps)
            patch = pylab.Circle(matlab_ps[0], radius=radius, color=color)
            ax.add_patch(patch)


def plot_transformed_image_and_keypoints(
    json_file=None, keypoints=None, transformation=None, doer="cv", display=False
):
    if json_file is None:
        json_file = load_test_json_file()

    img = get_image(json_file)
    oois = get_objects_of_interest(json_file)
    img_shape = np.array(img.shape[:2])
    oois_points = make_points_homogeneous(oois["points"])
    oois_labels_and_indices = oois["labels_and_indices"]

    if keypoints is None:
        keypoints = get_corners() * img_shape
        keypoints = make_points_homogeneous(keypoints)

    if transformation is None:
        transformation = get_random_transformation()

    print("transformation")
    print(transformation)
    # output_shape = get_output_shape(img_shape, transformation.params)
    transformed_keypoints = get_transformed_points(
        keypoints, transformation._inv_matrix
    )
    print(f"transformed_keypoints {(transformed_keypoints).astype(int)}")
    bbox = get_rectangle_from_polygon(
        transformed_keypoints[:, :2], encoding="preferred"
    )
    print(f"bbox {bbox}")

    output_shape = np.ceil(bbox[2:]).astype(int)

    print(f"output_shape {output_shape}")
    center = np.array(bbox[:2])
    center_shift = transformed_keypoints[-1, :2] - output_shape / 2

    print(f"center_shift {center_shift}")

    shift = ski.transform.AffineTransform(translation=center_shift[::-1])
    print(f"shift {shift}")

    final_transformation = shift + transformation
    print(f"final_transformation {final_transformation}")

    transformed_keypoints[:, :2] -= center_shift

    oois_points[:, :2] *= img_shape
    transformed_oois_points = get_transformed_points(
        oois_points, final_transformation._inv_matrix
    )
    # print(f'transformed_oois_points {transformed_oois_points}')
    transformed_image = get_transformed_image(
        img, final_transformation, output_shape=output_shape, doer=doer
    )

    fig, axes = pylab.subplots(1, 2)

    axes[0].imshow(img)
    axes[0].set_title("Original image")
    axes[0].set_axis_off()
    # plot_keypoints(keypoints, radius=7, ax=axes[0])
    plot_oois(oois_points[:, :2], oois_labels_and_indices, ax=axes[0])

    axes[1].imshow(transformed_image)
    axes[1].set_title("Transformed image")
    axes[1].set_axis_off()
    # plot_keypoints(transformed_keypoints, radius=7, ax=axes[1])
    plot_oois(transformed_oois_points[:, :2], oois_labels_and_indices, ax=axes[1])
    pylab.savefig("transform_%.1f.jpg" % time.time())
    if display:
        pylab.show()

    # matrix = transformation.params
    # matrix[2:,:2] += shift._inv_matrix[2:, :2]
    # alternative_final_transformation = ski.transform.AffineTransform(matrix=matrix)
    # print(f'alternative_final_transformation {alternative_final_transformation}')

    # https://docs.opencv.org/4.x/d4/d61/tutorial_warp_affine.html
    # srcTri = keypoints[:3, :2][:, ::-1].astype(np.float32)
    # dstTri = transformed_keypoints[:3, :2][:, ::-1].astype(np.float32)
    # print('src dst')
    # print(srcTri)
    # print(dstTri)

    # warp_mat = cv.getAffineTransform(srcTri, dstTri)
    # print(f'estimated warp_mat {warp_mat}')
    # matrix = np.eye(3)
    # matrix[:2, :] = warp_mat[:, :]

    # estimated_transformation = ski.transform.AffineTransform(matrix=np.linalg.inv(matrix))
    # print(f'estimated_transformation {estimated_transformation}')
    # print(f'estimated matrix')
    # print(f'{matrix}')
    # transformed_image = get_transformed_image(img, estimated_transformation, output_shape=output_shape, doer=doer)


def save_annotation_figure(
    annotation, alpha=0.25, dpi=192, factor=1.299, lw=1, masks=False
):
    json_file = load_json(annotation)
    image = get_image(json_file)
    fig = pylab.figure(figsize=np.array(image.shape[:2][::-1]) / dpi)
    pylab.imshow(image)

    if masks:
        hm = get_hierarchical_mask(json_file)
        pylab.imshow(hm, alpha=alpha)
    else:
        oois = get_objects_of_interest(json_file)
        fractional = oois["fractional"]
        image_shape = oois["image_shape"]

        patches = []
        for label, indices in zip(oois["labels"], oois["indices"]):
            if label in ["not_background"]:
                continue
            ls = "solid"
            if label in line_styles:
                ls = line_styles[label]
            if label in colors_for_labels:
                color = sns.xkcd_rgb[colors_for_labels[label]]
            elif label not in colors_for_labels and label in additional_labels:
                _label = additional_labels[label]
                color = sns.xkcd_rgb[colors_for_labels[_label]]
            else:
                print(f"label {label} not accounted for, please check")
                color = sns.xkcd_rgb[colors_for_labels["not_background"]]

            points = oois["points"][indices[0] : indices[1], :]

            if fractional:
                points = points * image_shape

            matlab_points = points[:, ::-1]
            if len(points) >= 3:
                patch = pylab.Polygon(
                    matlab_points, color=color, lw=lw, fill=False, ls=ls
                )
            elif len(points) == 2:
                print("Rectangle")
                x, y, width, height = get_rectangle_from_polygon(points)
                print("x, y, w, h", x, y, width, height)
                patch = pylab.Rectangle(
                    (y - height // 2, x - width // 2),
                    height,
                    width,
                    color=color,
                    lw=lw,
                    fill=False,
                    ls=ls,
                )
            elif len(points) == 1:
                patch = pylab.Circle(matlab_points[0], radius=7, color=color, ls=ls)

            patches.append((label, patch))

        ax = pylab.gca()
        patches.sort(key=lambda x: -notion_importance[x[0]])
        legends = []

        for label, patch in patches:
            ax.add_patch(patch)
            if label not in legends:
                print(f"setting legend for label {label}")
                legends.append(label)
                patch.set_label(label)

    print("all patches")
    for patch in patches:
        print(patch)

    pylab.axis("off")
    pylab.legend()
    fig.savefig(
        annotation.replace(".json", "_overview.jpg"),
        bbox_inches="tight",
        pad_inches=0,
        dpi=dpi * factor,
    )


def show_annotations(json_file):
    image = get_image(json_file)
    pylab.figure(figsize=(16, 9))
    pylab.title("Sample image", fontsize=22)
    ax = pylab.gca()
    pylab.axis("off")
    ax.imshow(image)
    # pylab.savefig('sample_image.jpg')
    pylab.show()
    pylab.figure(figsize=(16, 9))
    pylab.title("Segmentation map", fontsize=22)
    ax = pylab.gca()
    pylab.axis("off")
    ax.imshow(image)
    oois = get_objects_of_interest(json_file)
    hierarchical_mask = get_hierarchical_mask(json_file)
    pylab.imshow(hierarchical_mask, alpha=0.75)
    # pylab.savefig('hierarchical_mask.jpg')
    pylab.show()
    image_shape = oois["image_shape"]
    pylab.figure(figsize=(16, 9))
    pylab.title("All targets", fontsize=22)
    ax = pylab.gca()
    pylab.axis("off")
    ax.imshow(image)
    for label in oois:
        if label in [
            "image_shape",
            "labeled_points",
            "image",
            "labels",
            "points",
            "indices",
        ]:
            continue
        print("label", label)
        if label not in colors_for_labels and label in additional_labels:
            label = additional_labels[label]
        color = sns.xkcd_rgb[colors_for_labels[label]]
        coois = copy.deepcopy(oois)
        for points in coois[label]:
            points *= image_shape
            matlab_points = points[:, ::-1]
            print("points", points)
            if len(points) >= 3:
                patch = pylab.Polygon(matlab_points, color=color, lw=2, fill=False)
                ax.add_patch(patch)
                x, y, width, height = get_rectangle_from_polygon(points)
                patch = pylab.Rectangle(
                    (y - height // 2, x - width // 2),
                    height,
                    width,
                    color=color,
                    lw=2,
                    fill=False,
                )
                ax.add_patch(patch)

                # ax.imshow(mask, alpha=0.15)
                (r, c), (r_radius, c_radius), orientation = get_ellipse_from_polygon(
                    points
                )
                # print(
                # "ellipse",
                # r * image_shape[0],
                # c * image_shape[1],
                # r_radius * image_shape[0],
                # c_radius * image_shape[1],
                # orientation,
                # )
                # mask = get_mask_from_polygon(points, image_shape)
                # r, c, major, minor, orientation = get_ellipse_from_mask(mask)
                # print("ellipse", (r, c), major, minor, orientation)
                patch = matplotlib.patches.Ellipse(
                    (c, r),
                    c_radius,  # major
                    r_radius,  # minor
                    angle=-orientation,
                    color=color,
                    fill=False,
                    lw=2,
                )
                ax.add_patch(patch)

            elif len(points) == 1:
                patch = pylab.Circle(matlab_points[0], radius=7, color=color)
                ax.add_patch(patch)

    aoi = get_aoi_mask(oois)

    if np.any(aoi):
        print("aoi")
        # pylab.imshow(aoi, alpha=0.5)
        epo, epi, epooa, epioa, pa = get_extreme_point(aoi)
        patch = pylab.Circle(
            epooa[::-1], radius=7, color=sns.xkcd_rgb[colors_for_labels["extreme"]]
        )
        ax.add_patch(patch)
        patch = pylab.Circle(
            epioa[::-1], radius=7, color=sns.xkcd_rgb[colors_for_labels["start_likely"]]
        )
        ax.add_patch(patch)

    support = get_support_mask(oois)

    # pylab.figure()
    # pylab.imshow(support)
    if np.any(support):
        print("support")
        # ax.imshow(support, alpha=0.5)
        epo, epi, epooa, epioa, pa = get_extreme_point(support)
        patch = pylab.Circle(
            epioa[::-1],
            radius=7,
            color=sns.xkcd_rgb[colors_for_labels["start_possible"]],
        )
        ax.add_patch(patch)

    # all_but_pin = get_label_mask(oois, ['crystal', 'loop_inside', 'loop', 'stem', 'cd_loop', 'cd_stem'])
    pylab.savefig("all_together.jpg")
    pylab.show()


def load_json(
    fname="/nfs/data2/Martin/Research/murko/manually_segmented_images/json/spine/dls_i04/6116020_fullscreen-30086648_201.40800000000002.json",
):
    f = json.load(open(fname, "rb"))
    return f


def load_test_json_file():
    json_file = json.load(
        open(
            "soleil_proxima_dataset/100161_Wed_Jul_10_21:09:10_2019_double_click_zoom_2_y_529_x_606.json",
            "rb",
        )
    )
    return json_file


def load_test_image():
    json_file = load_test_json_file()
    img = get_image(json_file)
    return img


def main():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--json",
        default="soleil_proxima_dataset/100161_Wed_Jul_10_21:09:10_2019_double_click_zoom_2_y_529_x_606.json",
        type=str,
        help="path to the json file containing sample annotation",
    )
    args = parser.parse_args()
    print("args", args)
    # json_file = json.load(open(args.json, "rb"))
    # show_annotations(json_file)

    save_annotation_figure(args.json)


if __name__ == "__main__":
    # main()
    # translation = np.array([20, 45])
    # rotation = np.deg2rad(-15)

    # img = load_test_image()/255.

    # center = np.array(img.shape[:2])/2.

    # transformation = ski.transform.AffineTransform(rotation=rotation, translation=translation)
    # shift_c = ski.transform.AffineTransform(translation=-center)
    # shift_invc = ski.transform.AffineTransform(translation=+center)
    # final_transformation = shift_c + transformation + shift_invc

    # keypoints = get_corners() * np.array(img.shape[:2])
    # keypoints += np.array((200, 200))
    # keypoints = make_points_homogeneous(keypoints)

    # img = np.pad(img, pad_width=((200, 200), (200, 200), (0, 0)), constant_values=1)
    # plot_transformed_image_and_keypoints(json_file=None, keypoints=None, transformation=None, doer='cv')
    main()
    # json_file = load_test_json_file()
    # oois = get_objects_of_interest(json_file)
    # targets = get_targets(oois)
    ##print('targets')
    # print(targets)
