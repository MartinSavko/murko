#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import numpy as np
from scipy.spatial import distance_matrix
try:
    import peakutils
except:
    peakutils = None
import time

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

def get_gang_of_five(labels, indices, points, properties):
    
    lipp = (labels, indices, points, properties)
    _origin = get_origin(*lipp)
    
    sp = get_start_possible(*lipp + (_origin,))
    sl = get_start_likely(*lipp + (_origin,))
    extreme = get_extreme(*lipp + (_origin,))
    
    mlc = get_most_likely_click(*lipp + (extreme,))
    
    gang_of_five = {
        "origin": get_refined_origin(*lipp + (_origin,)),
        "most_likely_click": mlc,
        "start_likely": sl,
        "start_possible": sp,
        "extreme": extreme,
    }

    return gang_of_five



def get_origin(labels, indices, points, properties):
    origin = np.array((-1, -1))
    if "foreground" in labels:
        f = properties[labels.index("foreground")]
        
        a = properties[labels.index("aether")]
        b = f.get_dense_boundary()
        ac = np.array(a.get_inner_center())
        distances = np.linalg.norm(b - ac, axis=1)
        origin = b[np.argmax(distances)]

        #epoints = np.array(f.get_extreme_points())
        #origin = epoints[np.argmin(np.linalg.norm(epoints - origin, axis=1))]
        #origin = minmax["max_major"]
        #origin = get_minmax(f.get_mask())["max_major"]
        #o1, o2 = minmax["max_major"], minmax["max_minor"]
        minmax = get_minmax(f.get_mask())
        points = np.array(list(minmax.values()))
        distances = np.linalg.norm(points - origin, axis=1)
        origin = points[np.argmin(distances)]
    print(f"origin {origin}")
    return origin


def get_refined_origin(labels, indices, points, properties, origin=None, start_possible=None):
    lipp = (labels, indices, points, properties)
    if origin is None:
        origin = self._get_origin(*lipp)
    if "pin" in labels:
        if start_possible is None:
            start_possible = get_start_possible(*lipp + (origin,))
        projection = properties[labels.index("pin")].get_mask()
        named_pca_points = get_named_pca_points(start_possible, projection, origin_is_extreme=True)
        refined_origin = named_pca_points["bottom"]
    else:
        refined_origin = origin
    return refined_origin

def get_extreme(labels, indices, points, properties, origin=None):
    extreme = np.array((-1, -1))
    if "foreground" in labels:
        if origin is None:
            origin = _get_origin(labels, indices, points, properties)
        k = labels.index("foreground")
        b = properties[k].get_dense_boundary()
        distances = np.linalg.norm(b - origin, axis=1)
        extreme = b[np.argmax(distances)]

    return extreme

def get_origin_and_extreme(labels, indices, points, properties):
    extreme, origin = np.array((-1, -1)), np.array((-1, -1))
    
    if "foreground" in labels:
        f = properties[labels.index("foreground")]
        origin = get_minmax(f.get_mask())["max_major"]
        b = f.get_dense_boundary()
        distances = np.linalg.norm(b-origin, axis=1)
        extreme = b[np.argmax(distances)]
    return origin, extreme
    

def get_most_likely_click(labels, indices, points, properties, extreme=None):
    mlc = np.array((-1, -1))
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
        if extreme is None:
            extreme = get_extreme(labels, indices, points, properties)
        mlc = extreme
        
    return mlc


def get_start_possible(labels, indices, points, properties, origin=None):
    if origin is None:
        origin = get_origin(labels, indices, points, properties)
    sp = origin.copy()
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

    return sp


def get_start_likely(labels, indices, points, properties, origin=None):
    if origin is None:
        origin = get_origin(labels, indices, points, properties)
    sl = origin.copy()
    if "area_of_interest" in labels:
        a = properties[labels.index("area_of_interest")]
        b = a.get_dense_boundary()
        sl = b[np.argmin(np.linalg.norm(b - sl, axis=1))]
    return sl

def get_stem_center(labels, indices, points, properties):
    stem_center = np.array((-1, -1))
    if "stem" in labels:
        stem_center = properties[labels.index("stem")].get_inner_center()
    return stem_center

def get_ltrbc(labels, indices, points, properties, label="area_of_interest"):
    l = np.array((-1, -1))
    t = np.array((-1, -1))
    r = np.array((-1, -1))
    b = np.array((-1, -1))
    c = np.array((-1, -1))
    if label in labels:
        o = properties[labels.index(label)]
        c = o.get_inner_center()

        sl = get_origin(labels, indices, points, properties)
        epoints = np.array(o.get_eigen_points())
        el = list(epoints)
        
        distances = np.linalg.norm(np.array(el) - sl, axis=1)
        b = el.pop(np.argmin(distances))
        
        distances = np.linalg.norm(np.array(el) - b, axis=1)
        t = el.pop(np.argmax(distances))
        e1 = t - b

        v1 = el[0] - b
        v2 = el[1] - b
        if np.cross(e1, v1) > 0 and np.cross(e1, v2) <= 0:
            l = el[0]
            r = el[1]
        elif np.cross(e1, v1) <= 0 and np.cross(e1, v2) > 0:
            l = el[1]
            r = el[0]
        else:
            print(f"e1 {e1}, v1 {v1}, v2 {v2} should never get here, please check")
    
    return {"left": l, "top": t, "right": r, "bottom": b, "center": c}

def get_pin_right_and_left(
    labels, indices, points, properties, min_dist=0.25, filter_window=11
):
    r = np.array((-1, -1))
    l = np.array((-1, -1))
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


def get_orientation_and_direction(origin, extreme):
    # assuming origin and extreme points are in [V, H] format
    vector = extreme - origin
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

    minmax = {"center": center}
    for a, k in zip(("major", "minor"), (0, 1)):
        xyz_S_onaxis = xyz_S[np.isclose(xyz_S[:, abs(k - 1)], 0.0, atol=atol)]

        for l in ("min", "max"):
            key = f"{l}_{a}"
            try:
                minmax[key] = xyz_S_onaxis[getattr(np, f"arg{l}")(xyz_S_onaxis[:, k])]
            except:
                minmax[key] = xyz_S[getattr(np, f"arg{l}")(xyz_S[:, k])]

            minmax[key] = np.dot(minmax[key], S_inv) + center
    
    for key in minmax: 
        minmax[key] = minmax[key][::-1]

    return minmax


def get_named_pca_points_old(
    projection,
    orientation,
    direction,
    atol=5,
    default=np.array((-1, -1)),
    order=["top", "bottom", "left", "right"],
    #unit_cross=[[1.0, 0.0], [-1.0, 0.0], [0.0, -1.0], [0.0, 1.0]]
):

    minmax = get_minmax(projection, atol=atol)

    points = list(minmax.values())
    scaled = [a / np.linalg.norm(a) for a in points]
    dm = distance_matrix(get_oriented_unit_cross(orientation, direction), scaled)
    point_order = np.argmin(dm, axis=1)

    npp = {"center": minmax["center"]}
    for k, name in enumerate(order):
        npp[name] = points[point_order[k]]

    return npp


def get_named_pca_points(origin, projection, origin_is_extreme=False):
    
    minmax = get_minmax(projection)
    points = np.array(list(minmax.values()))
    
    distances = np.linalg.norm(points - origin, axis=1)
    
    keys = list(minmax.keys())
    
    if origin_is_extreme:
        nachalo = points[np.argmin(distances)]
    else:
        nachalo = points[np.argmin(distances)]
    
    nachalo_key = keys[np.squeeze(np.argwhere(np.all(np.isclose(nachalo, [minmax[key] for key in keys]), axis=1)))]
    
    if "max_" in nachalo_key:
        konec_key = nachalo_key.replace("max_", "min_")
    elif "min_" in nachalo_key:
        konec_key = nachalo_key.replace("min_", "max_")
    else:
        print("this should not have happened, is it that the center was found as a bottom, something off with the supplied origin?")
        print(f"minmax {minmax}")

    if origin_is_extreme:
        bottom_key, top_key = konec_key, nachalo_key
    else:
        bottom_key, top_key = nachalo_key, konec_key
    
    top = minmax[top_key]
    bottom = minmax[bottom_key]
    
    if "major" in bottom_key:
        tentative_left_key = bottom_key.replace("major", "minor")
        tentative_right_key = top_key.replace("major", "minor")
    elif "minor" in bottom_key:
        tentative_left_key = bottom_key.replace("minor", "major")
        tentative_right_key = top_key.replace("minor", "major")
    else:
        print("this should not have happened, is it that the center was found as a bottom, something off with the supplied origin?")
        print(f"minmax {minmax}")

    tr, tl = minmax[tentative_right_key], minmax[tentative_left_key]
    
    e = top - bottom

    vr = tr - bottom
    vl = tl - bottom
    if np.cross(e, vr) > 0 and np.cross(e, vl) <= 0:
        l = tl
        r = tr
        right_key = tentative_right_key
        left_key = tentative_left_key
    elif np.cross(e, vr) <= 0 and np.cross(e, vl) > 0:
        l = tr
        r = tl
        right_key = tentative_left_key
        left_key = tentative_right_key
    else:
        print(f"e {e}, vl {vl}, vr {vr} should never get here, please check")
        print("minmax")
        print(minmax)
        
    npp = {
        "center": minmax["center"],
        "bottom": minmax[bottom_key],
        "top": minmax[top_key],
        "left": minmax[left_key],
        "right": minmax[right_key],
    }

    return npp

    

    


