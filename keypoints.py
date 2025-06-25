#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import numpy as np
from scipy.spatial import distance_matrix
import peakutils

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


def get_pin_right_and_left(labels, indices, points, properties, min_dist=0.25, filter_window=11):
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

        indices = peakutils.indexes(-(d2-d1), min_dist=0.25*len(pin), thres=0.5)
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
