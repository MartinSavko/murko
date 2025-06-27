#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import numpy as np
from scipy.spatial import distance_matrix
import peakutils


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
    return xyz, inertia, eigenvalues, eigenvectors, center


def get_origin(labels, indices, points, properties):
    origin = (-1, -1)
    if "foreground" in labels:
        f = properties[labels.index("foreground")]
        a = properties[labels.index("aether")]
        fb = f.get_dense_boundary()
        ac = np.array(a.get_inner_center(pad=0))
        distances = np.linalg.norm(fb - ac, axis=1)
        origin = fb[np.argmax(distances)]

        epoints = np.array(f.get_extreme_points())

        origin = epoints[np.argmin(np.linalg.norm(epoints - origin, axis=1))]

    return origin


def get_extreme(labels, indices, points, properties):
    extreme = (-1, -1)
    if "foreground" in labels:
        origin = _get_origin(labels, indices, points, properties)
        k = labels.index("foreground")
        fb = properties[k].get_dense_boundary()
        distances = np.linalg.norm(fb - origin, axis=1)
        extreme = fb[np.argmax(distances)]

    return extreme


def get_most_likely_click(labels, indices, points, properties):
    mlc = (-1, -1)
    largest_area = -np.inf

    if "user_click" in labels:
        mlc = np.squeeze(_get_points("user_click", labels, indices, points)[:, ::-1])
    elif "crystal" in labels:
        for k, label in enumerate(labels):
            if label == "crystal":
                area = properties[k].get_area()
                if area > largest_area:
                    mlc = properties[k].get_inner_center()
                    largest_area = area
    elif "area_of_interest" in labels:
        k = labels.index("area_of_interest")
        mlc = properties[k].get_inner_center()
    elif "foreground" in labels:
        mlc = get_extreme(labels, indices, points, properties)

    return mlc


def get_start_possible(labels, indices, points, properties):
    sp = (-1, -1)
    if "support" in labels:
        s = properties[labels.index("support")]
        support = s.get_dense_boundary()
        if "pin" in labels:
            p = properties[labels.index("pin")]
            pin = p.get_dense_boundary()

            dm = distance_matrix(support, pin)
            sm = dm.min(axis=1)
            frontier = support[sm == sm.min()]
            sp = np.median(frontier, axis=0)
        else:
            sp = get_origin(labels, indices, points, properties)

    return sp


def get_start_likely(labels, indices, points, properties):
    sl = (-1, -1)
    if "area_of_interest" in labels:
        a = properties[labels.index("area_of_interest")]
        aoi = a.get_dense_boundary()
        if "stem" in labels:
            s = properties[labels.index("stem")]
            stem = s.get_dense_boundary()
            dm = distance_matrix(stem, aoi)
            sm = dm.min(axis=1)
            frontier = stem[sm == sm.min()]
            sl = np.median(frontier, axis=0)
        else:
            sl = get_origin(labels, indices, points, properties)

    return sl


def get_aoi_keypoints(labels, indices, points, properties, label="area_of_interest"):
    ab = (-1, -1)
    at = (-1, -1)
    al = (-1, -1)
    ar = (-1, -1)
    if label in labels:
        aoi = properties[labels.index(label)]
        # sl = get_start_likely(labels, indices, points, properties)
        sl = get_origin(labels, indices, points, properties)
        epoints = np.array(aoi.get_eigen_points())
        el = list(epoints)
        distances = np.linalg.norm(np.array(el) - sl, axis=1)
        ab = el.pop(np.argmin(distances))
        distances = np.linalg.norm(np.array(el) - ab, axis=1)
        at = el.pop(np.argmax(distances))
        e1 = at - ab

        v1 = el[0] - ab
        v2 = el[1] - ab
        if np.cross(e1, v1) > 0 and np.cross(e1, v2) <= 0:
            al = el[0]
            ar = el[1]
        elif np.cross(e1, v1) <= 0 and np.cross(e1, v2) > 0:
            al = el[1]
            ar = el[0]
        else:
            print(f"e1 {e1}, v1 {v1}, v2 {v2} should never get here, please check")
    return ab, at, al, ar


def get_pin_right_and_left(
    labels, indices, points, properties, min_dist=0.25, filter_window=11
):
    r = (-1, -1)
    l = (-1, -1)
    if "pin" in labels:
        p = properties[labels.index("pin")]
        pin = p.get_dense_boundary()

        # epoints = p.get_eigen_points()

        origin = get_origin(labels, indices, points, properties)
        sp = get_start_possible(labels, indices, points, properties)

        e1 = sp - origin

        d1 = np.linalg.norm(pin - origin, axis=1, ord=1) + np.linalg.norm(
            pin - sp, axis=1, ord=1
        )
        d2 = np.linalg.norm(pin - origin, axis=1, ord=2) + np.linalg.norm(
            pin - sp, axis=1, ord=2
        )

        indices = peakutils.indexes(-(d2 - d1), min_dist=0.25 * len(pin), thres=0.5)
        print(f"indices {indices}")
        p1 = pin[indices[0]]
        p2 = pin[indices[1]]

        v1 = p1 - origin
        v2 = p2 - origin

        if np.cross(e1, v1) > 0 and np.cross(e2, v2) <= 0:
            l = p2
            r = p1
        elif np.cross(e1, v1) <= 0 and np.cross(e2, v2) > 0:
            l = p1
            r = p2
        else:
            print(f"e1 {e1}, v1 {v1}, v2 {v2} should never get here, please check")
    return r, l


def _get_points(label, labels, indices, points):
    k = labels.index(label)
    idx = indices[k]
    return points[idx[0] : idx[1]]


def _get_origin(labels, indices, points, properties):
    if "origin" in labels:
        origin = _get_point("origin", labels, indices, points)
    else:
        origin = get_origin(labels, indices, points, properties)
    return origin


def _get_point(label, labels, indices, points, properties):
    if label in labels:
        point = _get_points(label, labels, indices, points)
    else:
        point = eval("get_{label}(labels, indices, points, properties)")
    return point


def draw_point(point, ax=None, radius=3, color="red"):
    if ax is None:
        ax = pylab.gca()

    p = pylab.Circle(point, radius=radius, color=color)
    ax.add_patch(p)


def get_orientation_and_direction(origin, extreme):
    # assuming origin and extreme points are in [V, H] format
    vector = global_extreme - global_origin
    theta = np.degrees(np.arctan2(vector[0], vector[1]))
    if (theta <= 45.0 and theta > -45.0) or (theta >= 135 and theta < -135):
        orientation = 1
    else:
        orientation = 0
    if vector[orientation] > 0:
        direction = +1
    else:
        direction = -1
    return orientation, direction


def get_oriented_unit_cross(
    orientation,  # 0 or 1; 0 for vertical, 1 for horizontal
    direction,  # 1 or -1; 1 for rising pixel number, -1 for decreasing pixel number
    unit_cross=[[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]],
):

    ouc = np.array(unit_cross)
    if orientation == 1:
        ouc = ouc[:, ::-1]

    if direction != 0:
        ouc[ouc != 0] = ouc[ouc != 0] * direction

    return ouc


def get_minmax(projection, atol=5):

    xyz, inertia, e, S, center = principal_axes(projection)

    S_inv = np.linalg.inv(S)
    xyz_O = xyz - center
    xyz_S = np.dot(xyz_O, S)

    minmax = {}
    for a, k in zip(("major", "minor"), (0, 1)):
        xyz_S_onaxis = xyz_S[np.isclose(xyz_S[:, abs(k - 1)], 0.0, atol=atol)]

        for l in ("min", "max"):
            key = f"{l}_{a}"
            try:
                minmax[key] = xyz_S_onaxis[getattr(np, f"arg{l}")(xyz_S_onaxis[:, k])]
            except:
                minmax[key] = xyz_S[getattr(np, f"arg{l}")(xyz_S[:, k])]

            minmax[key] = np.dot(minmax[key], S_inv) + center

    return minmax


def get_named_pca_points(
    projection,
    orientation,
    direction,
    atol=5,
    default=np.array((-1, -1)),
    order=["top", "bottom", "left", "right"],
):

    minmax = get_minmax(projection, atol=atol)

    ouc = get_oriented_unit_cross(orientation, direction)
    points = [item[key] for key in minmax]
    scaled = [a / np.linalg.norm(a) for a in points]
    dm = distance_matrix(ouc, scaled)
    point_order = np.argmin(dm, axis=1)

    npp = {"center": center}
    for k, name in enumerate(order):
        npp[name] = points[point_order[k]]

    return npp
