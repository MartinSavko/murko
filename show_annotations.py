#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import json
import numpy as np
import pylab
import matplotlib.patches
import seaborn as sns
from skimage.draw import polygon2mask
from labelme import utils
import time
import copy
import skimage

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
    "not_background": "greyish",
    "pin": "dusk blue",
    "stem": "crimson",
    "loop": "faded green",
    "loop_inside": "moss green",
    "ice": "custard",
    "dust": "cool grey",
    "capillary": "faded blue",
    "crystal": "banana yellow",
    "drop": "orangeish",
    "support": "dusk blue",
    "user_click": "banana yellow",
    "extreme": "dark aquamarine",
    "start_likely": "coral",
    "start_possible": "crimson",
}

additional_labels = {
    "cd_loop": "loop",
    "cd_stem": "stem",
}


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
    image = utils.img_b64_to_arr(json_file.get("imageData"))
    return image


def get_shapes(json_file):
    shapes = json_file.get("shapes")
    return shapes


def get_image_shape(json_file):
    image_shape = np.array((json_file.get("imageHeight"), json_file.get("imageWidth")))
    return image_shape


@timeit
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

@timeit
def get_label_mask(oois, labels):
    image_shape = oois["image_shape"]
    label_mask = np.zeros(image_shape, dtype=np.uint8)
    for label in labels:
        if label not in oois:
            continue

        for points in oois[label]:
            if len(points) < 3:
                continue
            print("label", label)
            polygon = points * image_shape
            mask = get_mask_from_polygon(polygon, image_shape)
            print("type(mask), dtype", type(mask), mask.dtype)
            label_mask = np.logical_or(label_mask == 1, mask == 1)
    return label_mask


def get_complex_mask(oois, positive=["crystal", "loop_inside", "loop"], negative=[]):
    positive_mask = get_label_mask(oois, positive)
    negative_mask = get_label_mask(oois, negative)
    complex_mask = np.logical_xor(positive_mask, negative_mask)
    return complex_mask


def get_aoi(oois):
    aoi = get_label_mask(oois, ["crystal", "loop_inside", "loop", "cd_loop"])
    return aoi


def get_support(oois):
    positive = get_label_mask(oois, ["stem", "loop", "cd_loop", "cd_stem"])
    negative = get_label_mask(oois, ["loop_inside"])
    support = np.logical_xor(positive, negative)
    return support


@timeit
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


@timeit
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


@timeit
def get_ellipse_from_polygon(polygon):
    # https://stackoverflow.com/questions/47873759/how-to-fit-a-2d-ellipse-to-given-points
    Y = polygon[:, 0:1]
    X = polygon[:, 1:]
    A = np.hstack([X**2, X * Y, Y**2, X, Y])
    print("A.shape", A.shape)
    print("A")
    print(A)
    b = np.ones_like(X)
    x = np.linalg.lstsq(A, b)
    print("solution ", x)
    solution = x[0].squeeze()
    return solution


@timeit
def get_rps(mask):
    rps = skimage.measure.regionprops(skimage.measure.label(mask))[0]
    return rps

@timeit
def get_ellipse_from_mask(mask):
    rps = get_rps(mask)
    r, c = rps.centroid
    major = rps.axis_major_length
    minor = rps.axis_minor_length
    orientation = rps.orientation
    return r, c, major, minor, orientation

@timeit
def get_ellipse_from_rps(rps):
    r, c = rps.centroid
    major = rps.axis_major_length
    minor = rps.axis_minor_length
    orientation = rps.orientation
    return r, c, major, minor, orientation

@timeit
def get_mask_from_polygon(polygon, image_shape=(1200, 1600)):
    mask = polygon2mask(image_shape, polygon)
    return mask


@timeit
def get_objects_of_interest(json_file):
    shapes = get_shapes(json_file)
    image_shape = get_image_shape(json_file)
    objects_of_interest = {"image_shape": image_shape}
    
    points = []
    labels = []
    i_start = 0
    for shape in shapes:
        label = shape["label"]
        ooi = np.array(shape["points"])
        ooi = ooi[:, ::-1]  # swap x and y
        ooi /= image_shape
        if label not in objects_of_interest:
            objects_of_interest[label] = []
        objects_of_interest[label].append(ooi)
        points = np.vstack([points, ooi]) if points != [] else ooi
        i_end = i_start + ooi.shape[0]
        labels.append((label, i_start, i_end))
        i_start = i_end
    labeled_points = {'labels': labels, 'points': points}
    objects_of_interest['labeled_points'] = labeled_points
    return objects_of_interest

@timeit
def get_rectangle(bbox, encoding="matplotlib"):
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

@timeit
def get_rectangle_from_rps(rps, encoding="matplotlib"):
    rectangle = get_rectangle(rps.bbox, encoding=encoding)
    return rectangle
     
@timeit
def get_rectangle_from_polygon(polygon, encoding="matplotlib"):
    pvmax = polygon[:, 0].max()
    pvmin = polygon[:, 0].min()
    phmax = polygon[:, 1].max()
    phmin = polygon[:, 1].min()
    bbox = [pvmin, phmin, pvmax, phmax]
    rectangle = get_rectangle(bbox, encoding=encoding)
    return rectangle

@timeit
def get_support_mask(oois):
    support = get_label_mask(oois, ["loop", "stem", "cd_loop", "cd_stem"]).astype(int)
    nsupport = get_label_mask(oois, ["loop_inside"])
    support_mask = np.logical_xor(support, nsupport)
    return support_mask

@timeit
def get_aoi_mask(oois):
    aoi_mask = get_label_mask(oois, ["crystal", "loop_inside", "loop", "cd_loop", "cd_stem"])
    return aoi_mask

def show_annotations(json_file):
    image = get_image(json_file)
    pylab.figure(figsize=(16, 9))
    pylab.title('Sample image', fontsize=22)
    ax = pylab.gca()
    pylab.axis("off")
    ax.imshow(image)
    #pylab.savefig('sample_image.jpg')
    pylab.show()
    pylab.figure(figsize=(16, 9))
    pylab.title('Segmentation map', fontsize=22)
    ax = pylab.gca()
    pylab.axis("off")
    ax.imshow(image)
    oois = get_objects_of_interest(json_file)
    hierarchical_mask = get_hierarchical_mask(json_file)
    pylab.imshow(hierarchical_mask, alpha=0.75)
    #pylab.savefig('hierarchical_mask.jpg')
    pylab.show()
    image_shape = oois["image_shape"]
    pylab.figure(figsize=(16, 9))
    pylab.title('All targets', fontsize=22)
    ax = pylab.gca()
    pylab.axis("off")
    ax.imshow(image)
    for label in oois:
        if label in ["image_shape", "labeled_points"]:
            continue
        print("label", label)
        if label not in colors_for_labels and label in additional_labels:
            label = additional_labels[label]
        color = sns.xkcd_rgb[colors_for_labels[label]]
        coois = copy.deepcopy(oois)
        for points in coois[label]:
            points *= image_shape
            matlab_points = points[:, ::-1]
            if len(points) >= 3:
                patch = pylab.Polygon(matlab_points, color=color, lw=2, fill=False)
                ax.add_patch(patch)
                x, y, width, height = get_rectangle_from_polygon(points)
                patch = pylab.Rectangle(
                    (x, y), width, height, color=color, lw=2, fill=False
                )
                ax.add_patch(patch)
                mask = get_mask_from_polygon(points, image_shape)
                #ax.imshow(mask, alpha=0.15)
                r, c, r_radius, c_radius, orientation = get_ellipse_from_polygon(points)
                print ('ellipse', r*image_shape[0], c*image_shape[1], r_radius*image_shape[0], c_radius*image_shape[1], orientation)
                r, c, major, minor, orientation = get_ellipse_from_mask(mask)
                print("ellipse", (r, c), major, minor, orientation)
                patch = matplotlib.patches.Ellipse(
                    (c, r),
                    major,
                    minor,
                    angle=-np.degrees(orientation - np.pi / 2),
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
    pylab.savefig('all_together.jpg')
    pylab.show()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--json",
        default="double_clicks_100161_Wed_Jul_10_210910_2019_double_click_zoom_2_y_529_x_606.json",
        type=str,
        help="path to the json file containing sample annotation",
    )
    args = parser.parse_args()
    print("args", args)
    json_file = json.load(open(args.json, "rb"))
    show_annotations(json_file)
