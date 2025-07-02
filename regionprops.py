#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import cv2 as cv
import numpy as np

class Regionprops(object):
    def __init__(self, points, image_shape=None, distance_transform_pad=2):
        self.points = points[:, ::-1]  # assuming input is vxh, cv works in hxv
        self.image_shape = tuple(image_shape)
        self.bbox = None
        self.bbox_mask = None
        self.bbox_points = None
        self.bbox_extent = None
        self.bbox_center = None
        self.centroid = None
        self.min_rectangle = None
        self.min_enclosing_circle = None
        self.ellipse = None
        self.area = None
        self.perimeter = None
        self.moments = None
        self.mask = None
        self.mask_points = None
        self.distance_transform = None
        self.centerness = None
        self.inner_center = None
        self.blank = None
        self.bbox_ltrb = None
        self.ltrb_boundary = None
        self.get_bbox_as_minbox = None
        self.dense_boundary = None
        self.aspect = None
        self.eigen_points = None
        self.extreme_points = None
        self.extent = None
        self.solidity = None
        self.chebyshev = None
        self.boundary_interpolator = None
        self.chebyshev_basis = None
        self.thetas = None
        self.sph_coeff = None
        self.distance_transform_pad = distance_transform_pad

    def get_blank(self, image_shape=None):
        if image_shape is not None:
            image_shape = image_shape
        else:
            image_shape = self.image_shape
        blank = np.zeros(image_shape, np.uint8)
        return blank

    def _get_mask(self, points, image_shape, pad=0):
        ims = (image_shape[0] + 2 * pad, image_shape[1] + 2 * pad)
        mask = cv.fillPoly(self.get_blank(ims), [points.astype(np.int32) + pad], 1)
        return mask

    def _check_points_and_shape(self, points, image_shape):
        points = points if points is not None else self.points
        image_shape = image_shape if image_shape is not None else self.image_shape
        return points, image_shape

    # @saver
    def get_mask(self, points=None, image_shape=None, pad=0):
        points, image_shape = self._check_points_and_shape(points, image_shape)
        self.mask = self._get_mask(self.points, image_shape, pad=pad)
        return self.mask

    # @saver
    def get_mask_points(self):
        self.mask_points = np.argwhere(self.get_mask().astype(bool))
        return self.mask_points

    def _get_distance_transform(self, points=None, image_shape=None, exagerate=False, invert=False, power=1, pad=0):
        points, image_shape = self._check_points_and_shape(points, image_shape)
        _mask = self._get_mask(points, image_shape, pad=pad)
        distance_transform = get_distance_transform(_mask.astype('uint8'), exagerate=exagerate, invert=invert, power=power)
        if pad > 0:
            distance_transform = distance_transform[pad: -pad, pad: -pad]
        return distance_transform
    
    # @saver
    def get_distance_transform(self, points=None, image_shape=None, exagerate=False, invert=False, power=1):
        distance_transform = self._get_distance_transform(points=points, image_shape=image_shape, pad=self.distance_transform_pad, exagerate=exagerate, invert=invert, power=power)
        return distance_transform
    
    def get_inverse_distance_transform(self, points=None, image_shape=None, exagerate=False, invert=True, power=1):
        distance_transform = self._get_distance_transform(points=points, image_shape=image_shape, pad=self.distance_transform_pad, exagerate=exagerate, invert=invert, power=power)
        return distance_transform
    
    
    def get_sqrt_distance_transform(self, points=None, image_shape=None, exagerate=True, invert=False, power=0.5):
        distance_transform = self._get_distance_transform(points=points, image_shape=image_shape, pad=self.distance_transform_pad, exagerate=exagerate, invert=invert, power=power)
        return distance_transform
    
    def get_sqrt_inverse_distance_transform(self, points=None, image_shape=None, exagerate=True, invert=True, power=0.5):
        distance_transform = self._get_distance_transform(points=points, image_shape=image_shape, pad=self.distance_transform_pad, exagerate=exagerate, invert=invert, power=power)
        return distance_transform
    
    def get_power_distance_transform(self, points=None, image_shape=None, exagerate=True, invert=False, power=4):
        distance_transform = self._get_distance_transform(points=points, image_shape=image_shape, pad=self.distance_transform_pad, exagerate=exagerate, invert=invert, power=power)
        return distance_transform
    
    def get_power_inverse_distance_transform(self, points=None, image_shape=None, exagerate=True, invert=True, power=4):
        distance_transform = self._get_distance_transform(points=points, image_shape=image_shape, pad=self.distance_transform_pad, exagerate=exagerate, invert=invert, power=power)
        return distance_transform
    
    
    def get_centerness(self, points=None, image_shape=None):
        points, image_shape = self._check_points_and_shape(points, image_shape)
        try:
            inner_center = self.get_inner_center()
            xv, yv = np.meshgrid(np.arange(imgage_shape[1]), np.arange(image_shape[0]))
            centerness = np.sqrt(
                (inner_center[0] - yv) ** 2 + (inner_center[1] - xv) ** 2
            )
            centerness = centerness / centerness.max()
            centerness = (1 - centerness) ** 2
        except:
            centerness = np.zeros(image_shape, dtype=np.float32)
        return centerness

    # @saver
    def get_bbox(self):
        # self.bbox = get_rectangle_from_polygon(self.points) #
        # x, y , w, h (top-left coordinate, width, height)
        self.bbox = cv.boundingRect(self.points)
        return self.bbox

    # @saver
    def get_bbox_as_minbox(self):
        center = self.get_bbox_center()
        extent = self.get_bbox_extent()
        self.get_bbox_as_minbox = (center, extent, 0)
        return self.get_bbox_as_minbox

    # @saver
    def get_bbox_points(self):
        center, extent, orientation = self.get_bbox_as_minbox()
        self.bbox_points = cv.boxPoints((center, extent, orientation))
        return self.bbox_points

    # @saver
    def get_bbox_mask(self, image_shape=None):
        if image_shape is None:
            image_shape = self.image_shape
        self.bbox_mask = self._get_mask(self.get_bbox_points(), image_shape)
        return self.bbox_mask

    def get_min_rectangle_mask(self, image_shape=None):
        if image_shape is None:
            image_shape = self.image_shape
        min_rectangle_points = cv.boxPoints(self.get_min_rectangle)
        self.bbox_mask = self._get_mask(min_rectangle_points, image_shape)
        return self.bbox_mask

    def get_ellipse_contour(self, ellipse=None):
        if ellipse is None:
            ellipse = self.get_ellipse
        ellipse_contour = cv.ellipse2Poly(ellipse)
        return ellipse_contour

    def get_ellipse_mask(self, ellipse=None, ellipse_contour=None, image_shape=None):
        if ellipse_contour is None:
            ellipse_contour = self.get_ellipse_contour(ellipse=ellipse)
        self.ellipse_mask = self._get_mask(ellipse_contour)
        return self.ellipse_mask

    # @saver
    def get_bbox_extent(self):
        x, y, w, h = self.get_bbox()
        self.bbox_extent = (w, h)
        return self.bbox_extent

    # @saver
    def get_bbox_center(self):
        # x, y, w, h (top-left coordinate (x, y), width, height)
        x, y, w, h = self.get_bbox()
        cx = x + w / 2
        cy = y + h / 2
        self.bbox_center = (cx, cy)
        return self.bbox_center

    # @saver
    def get_centroid(self):
        self.centroid = np.mean(self.get_mask_points(), axis=0)[::-1]
        return self.centroid

    # #@saver
    def get_inner_center(
        self, points=None, image_shape=None, method="medianmax",
    ):
        points, image_shape = self._check_points_and_shape(points, image_shape)
        dt = self.get_distance_transform(points, image_shape, pad=self.distance_transform_pad)
        if method == "medianmax":
            # https://stackoverflow.com/questions/17568612/how-to-make-numpy-argmax-return-all-occurrences-of-the-maximum
            indices = np.vstack(
                np.unravel_index(np.flatnonzero(dt == dt.max()), dt.shape)
            ).T
            imean = indices.mean(axis=0)
            idist = np.linalg.norm(indices - imean, axis=1)
            winner = indices[np.argmin(idist)]
        elif method == "firstmax":
            winner = np.unravel_index(np.argmax(dt), dt.shape)
        elif method == "maximum_position":
            winner = scipy.ndimage.maximum_position(dt)
        ic = tuple(winner[::-1])
        print("inner_center", ic)
        self.inner_center = ic
        return self.inner_center

    # #@saver
    def get_dense_boundary(self, points=None, image_shape=None, pad=2):
        points, image_shape = self._check_points_and_shape(points, image_shape)
        _mask = self._get_mask(points, image_shape, pad=self.distance_transform_pad)
        contours, _ = cv.findContours(
            _mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )
        shape = contours[0].shape
        db = np.reshape(contours[0], (shape[0], shape[-1]))
        self.dense_boundary = db - pad
        return self.dense_boundary

    def get_mask_boundary(self, mask):
        contours, _ = cv.findContours(
            mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_NONE
        )
        shape = contours[0].shape
        db = np.reshape(contours[0], (shape[0], shape[-1]))
        mask_boundary = db - pad
        return mask_boundary

    # @saver
    def get_ltrb_boundary(self):
        self.ltrb_boundary = np.zeros(self.image_shape + (4,), np.float32)

        print(f"self.ltrb_boundary shape {self.ltrb_boundary.shape}")
        dense_boundary = self.get_dense_boundary()

        L, T, R, B = [], [], [], []
        for x in sorted(list(set(dense_boundary[:, 0])))[1:-1]:
            ys = dense_boundary[dense_boundary[:, 0] == x]
            t = ys.min()
            b = ys.max()
            T.append(t)
            B.append(b)
        for y in sorted(list(set(dense_boundary[:, 1])))[1:-1]:
            xs = dense_boundary[dense_boundary[:, 1] == y]
            l = xs.min()
            r = xs.max()
            L.append(l)
            R.append(r)

        mask = self.get_mask().astype(bool)
        x = np.arange(0, self.image_shape[1], 1)
        y = np.arange(0, self.image_shape[0], 1)
        xv, yv = np.meshgrid(x, y)
        print(f"xv {xv}")
        print(f"yv {yv}")
        for k, boundary in enumerate((L, T, R, B)):
            boundary = np.array(boundary)
            print(f"{k}, boundary.shape: {boundary.shape}")
            bb = self.get_blank().astype(np.float32)
            print(f"bb.shape {bb.shape}")
            if k % 2 == 0:
                print(f"xv[mask].shape {xv[mask].shape}")
                # bb[mask] = xv[mask][:, 1] - boundary

                # np.abs(
                # np.apply_along_axis(lambda x: x - boundary, 1, xv[mask])
                # )
            else:
                print(f"yv[mask].shape {yv[mask].shape}")
                # bb[mask] = yv[mask][:, 0] - boundary
                # np.abs(
                # np.apply_along_axis(lambda y: y - boundary, 0, yv[mask])
                # )
            self.ltrb_boundary[:, :, k] = bb
        return self.ltrb_boundary

    # @saver
    def get_bbox_ltrb(self):
        self.bbox_ltrb = np.zeros(self.image_shape + (4,), np.float32)
        l, t, w, h = self.get_bbox()
        r, b = l + w, t + h
        bbox_mask = self.get_bbox_mask().astype(bool)
        x = np.arange(0, self.image_shape[1], 1)
        y = np.arange(0, self.image_shape[0], 1)
        xv, yv = np.meshgrid(x, y)
        for k, boundary in enumerate((l, t, r, b)):
            bb = self.get_blank().astype(np.float32)
            if k % 2 == 0:
                bb[bbox_mask] = np.abs(xv[bbox_mask] - boundary)
            else:
                bb[bbox_mask] = np.abs(yv[bbox_mask] - boundary)
            self.bbox_ltrb[:, :, k] = bb

        return self.bbox_ltrb

    def get_universal_ltrb(self, kind="mask"):
        self.universal_ltrb = np.zeros(self.image_shape + (4,), np.float32)

        mask = getattr(self, f"get_{kind}")()
        boundary = get_mask_boundary(mask)
        l, t, r, b = [], [], [], []
        for point in boundary:
            x, y = point
            if np.all(x <= boundary[boundary == y]):
                l.append(point)
            else:
                r.appdn(point)
            if np.all(y >= boundary[boundary == x]):
                t.append(point)
            else:
                b.append(point)

        x = np.arange(0, self.image_shape[1], 1)
        y = np.arange(0, self.image_shape[0], 1)
        xv, yv = np.meshgrid(x, y)
        for k, boundary in enumerate((l, t, r, b)):
            bb = self.get_blank().astype(np.float32)
            if k % 2 == 0:
                bb[bbox_mask] = np.abs(xv[bbox_mask] - boundary)
            else:
                bb[bbox_mask] = np.abs(yv[bbox_mask] - boundary)
            self.bbox_ltrb[:, :, k] = bb

        return self.universal_ltrb

    # @saver
    def get_ellipse(self):
        # (cx, cy), (MA, ma), angle
        self.ellipse = cv.fitEllipse(self.get_mask_points())
        return self.ellipse

    # @saver
    def get_min_rectangle(self):
        # (cx, cy), (w, h), angle
        self.min_rectangle = cv.minAreaRect(self.points)
        return self.min_rectangle

    # @saver
    def get_min_enclosing_circle(self):
        self.min_enclosing_circle = cv.minEnclosingCircle(self.points)
        return self.min_enclosing_circle

    # @saver
    def get_area(self):
        self.area = cv.contourArea(self.points)
        return self.area

    def get_perimeter(self):
        self.perimeter = cv.arcLength(self.points, True)
        return self.perimeter

    # @saver
    def get_moments(self):
        self.moments = cv.moments(self.get_mask_points())
        return self.moments

    # @saver
    def get_extreme_points(self):
        self.extreme_points = get_extreme_points(self.get_dense_boundary())
        return self.extreme_points

    # @saver
    def get_eigen_points(self):
        # mask = self.get_mask()
        self.eigen_points = get_eigen_points(self.get_dense_boundary())
        return self.eigen_points

    # @saver
    def get_aspect(self):
        x, y, w, h = self.get_bbox()
        self.aspect = float(w) / h
        return self.aspcet

    # @saver
    def get_extent(self):
        area = self.get_area()
        x, y, w, h = self.get_bbox()
        rect_area = w * h
        self.extent = float(area) / rect_area
        return self.extent

    # @saver
    def get_solidity(self):
        area = self.get_area()
        hull = cv.convexHull(cnt)
        hull_area = cv.contourArea(hull)
        self.solidity = float(area) / hull_area
        return self.solidity

    # @saver
    def get_chebyshev_basis(self, n=20):
        self.chebyshev_basis = get_chebyshev_basis(n)
        return self.chebyshev_basis

    # @saver
    def get_boundary_interpolator(self, method="rbf"):

        tap = self.get_thetas_and_points()

        if method == "cs":
            self.boundary_interpolator = CubicSpline(
                tap[:, 0],
                tap[:, -1],
                # bc_type="periodic",
            )
        else:
            self.boundary_interpolator = RBFInterpolator(tap[:, 1, 2], tap[:, -1])
        return self.boundary_interpolator

    def get_thetas_and_points(
        self, center=None, ensure_monotonic=True, epsilon=1.0e-5, dense=True
    ):

        if center is None:
            center = np.array(self.get_inner_center())

        if dense or self.points.shape[0] < 21:
            points = self.get_dense_boundary()
        else:
            points = self.points

        points = points - center

        rs = np.linalg.norm(points, axis=1)

        xs = points[:, 0]
        ys = points[:, 1]

        thetas = np.arctan2(ys, xs)

        tap = list(zip(thetas, xs, ys, rs))
        tap.sort(key=lambda x: (x[0], -x[-1]))

        tap = np.array(tap)

        if ensure_monotonic:
            # we will identify thetas where there is more than just a single value and take the sample corresponding to the largest r value
            tap = make_tap_monotonic(tap, epsilon=epsilon)

        return tap

    def get_chebyshev_basis(
        self, degree=19, extend=True, method="numpy", npoints=401, domain=[-1.05, 1.05]
    ):
        tap = self.get_thetas_and_points()
        t = tap[:, 0]
        r = tap[:, -1]
        if extend:
            # hack to account for periodicity
            # start = tap[:extend]
            # end = tap[-extend:]
            # tap = np.vstack((end, tap))
            # tap = np.vstack((tap, start))
            tp = np.hstack([t, t + 2 * np.pi])
            rp = np.hstack([r, r])
            tp = np.hstack([t - 2 * np.pi, tp])
            rp = np.hstack([r, rp])
            t = tp[:]
            r = rp[:]

        interp = interp1d(t, r)
        tinterp = np.pi * np.linspace(domain[0], domain[1], npoints)
        rinterp = interp(tinterp)

        basis = get_chebyshev_basis(degree, points=tinterp)
        return basis

    # @saver
    def get_chebyshev(
        self, degree=19, extend=True, method="numpy", npoints=401, domain=[-1.05, 1.05]
    ):
        tap = self.get_thetas_and_points()
        t = tap[:, 0]
        r = tap[:, -1]
        if extend:
            tp = np.hstack([t, t + 2 * np.pi])
            rp = np.hstack([r, r])
            tp = np.hstack([t - 2 * np.pi, tp])
            rp = np.hstack([r, rp])
            t = tp[:]
            r = rp[:]

        interp = interp1d(t, r)
        tinterp = np.pi * np.linspace(domain[0], domain[1], npoints)
        rinterp = interp(tinterp)

        if method == "numpy":
            # https://www.oislas.com/blog/chebyshev-polynomials-fitting/
            x = np.polynomial.chebyshev.chebfit(tinterp, rinterp, degree)
        else:
            basis = get_chebyshev_basis(degree, points=tinterp)
            x, residuals, rank, s = np.linalg.lstsq(basis, rinterp, rcond=None)

        self.chebyshev = x

        return self.chebyshev

    # @saver
    def get_sph_coeff(self, degree=5, basis=None, tap=None):
        if tap is None:
            tap = self.get_thetas_and_points()
        thetas = tap[:, 0]
        rs = tap[:, -1]
        if basis is None:
            basis = get_spherical_basis(degree, points=thetas)
        x, residuals, rank, s = np.linalg.lstsq(basis, rs, rcond=None)
        self.sph_coeff = np.matrix(x).T
        return self.sph_coeff

    # @saver
    def get_thetas(self, domain=(-1, 1), npoints=401):
        self.thetas = np.pi * np.linspace(domain[0], domain[1], npoints)
        return self.thetas

    def get_boundary_from_chebyshev(
        self,
        coeff=None,
        center=None,
        thetas=None,
        method="numpy",
        domain=(-1.05, 1.05),
        npoints=401,
        degree=19,
    ):
        if thetas is None:
            thetas = self.get_thetas(domain=domain, npoints=npoints)
        if coeff is None:
            coeff = self.get_chebyshev(degree=degree, domain=domain, npoints=npoints)
        if center is None:
            center = np.array(self.get_inner_center())

        if method == "numpy":
            rs = np.polynomial.chebyshev.chebval(thetas, coeff)
        else:
            order = len(coeff) - 1
            basis = get_chebyshev_basis(order, points=thetas)
            rs = np.dot(basis, coeff)

        boundary = get_boundary_from_thetas_rs_and_center(thetas, rs, center)

        return boundary

    def get_boundary_from_spherical(self, coeff=None, center=None, thetas=None):
        if thetas is None:
            thetas = self.get_thetas()
        if coeff is None:
            coeff = self.get_sph_coeff()
        if center is None:
            center = np.array(self.get_inner_center())

        basis = get_spherical_basis(degree, points=thetas)

        rs = np.dot(basis, coeff)

        boundary = get_boundary_from_thetas_rs_and_center(thetas, rs, center)

        return boundary


def get_distance_transform(
    mask,
    normalize=True,
    distanceType=cv.DIST_L2,
    maskSize=3,
    power=1,
    invert=False,
    exagerate=False,
):
    dt = cv.distanceTransform(mask, distanceType, maskSize)
    cv.normalize(dt, dt, 0, 1, cv.NORM_MINMAX)
    if invert:
        dt = 1 - dt
        #dt[mask == 0] = 0
    if exagerate:
        dt = dt**power
    #if normalize:
        #cv.normalize(dt, dt, 0, 1, cv.NORM_MINMAX)
    return dt


def get_boundary_from_thetas_rs_and_center(thetas, rs, center):
    xs = np.cos(thetas) * rs
    ys = np.sin(thetas) * rs
    boundary = center + np.vstack([xs, ys]).T
    return boundary


def make_tap_monotonic(tap, epsilon=1.0e-5):

    thetas = tap[:, 0]
    t0 = thetas[:-1]
    t1 = thetas[1:]

    left_differences = t1 - t0

    monotonic = np.argwhere(left_differences > epsilon)

    tap = tap[monotonic.flatten()]

    # thetas = thetas[monotonic].flatten()
    # rs = rs[monotonic].flatten()

    # thetas = np.hstack([thetas, [thetas[0] + 2*np.pi]])
    # rs = np.hstack([rs, [rs[0]]])
    return tap


def get_spherical_basis(
    degree_max,
    order_max=5,
    degree_step=1,
    order_step=2,
    fname="sph_harm_y",
    domain=[0, 1],
    points=None,
    npoints=361,
    normalize=False,
    theta=0.5 * np.pi,
):
    if points is None:
        points = np.pi * np.linspace(domain[0], domain[1], npoints)

    phis = points
    basis = np.matrix(
        [
            getattr(scipy.special, fname)(degree, order, theta, phis)
            for degree in range(0, degree_max + 1, degree_step)
            for order in range(
                -min(order_max + degree % 2, degree),
                min(order_max + degree % 2, degree) + 1,
                order_step,
            )
        ]
    ).T
    # a = [(n, m) for n in range(0, 20+1, 2) for m in  range(-min(2, n), min(2, n) + 1, 2)]
    if normalize:
        basis = basis / np.linalg.norm(basis, axis=0)
    return basis


def get_spherical_basis2(
    degree_max,
    fname="sph_harm",
    domain=[0, 1],
    points=None,
    npoints=361,
    normalize=False,
    phi=0.5 * np.pi,
):
    if points is None:
        points = 2 * np.pi * np.linspace(domain[0], domain[1], npoints)

    thetas = points
    basis = np.matrix(
        [
            getattr(scipy.special, fname)(order, degree, thetas, phi)
            for degree in range(0, degree_max + 1)
            for order in range(-n, n + 1)
        ]
    ).T

    if normalize:
        basis = basis / np.linalg.norm(basis, axis=0)
    return basis


def get_chebyshev_basis(
    degree, domain=[-1, 1], points=None, npoints=361, typ="t", normalize=False
):
    if points is None:
        points = np.linspace(domain[0], domain[1], npoints)

    basis = np.matrix(
        [
            getattr(scipy.special, f"eval_cheby{typ}")(n, points)
            for n in range(0, degree + 1)
        ]
    ).T
    if normalize:
        basis = basis / np.linalg.norm(basis, axis=0)
    return basis


def get_point_line_distance(point, l1, l2, method="fast"):
    if method == "slow":
        # https://stackoverflow.com/questions/39840030/distance-between-point-and-a-line-from-two-points
        # 19.9 µs ± 112 ns per loop (mean ± std. dev. of 7 runs, 10,000 loops each)
        distance = np.cross(l1 - l2, l1 - point) / np.linalg.norm(l1 - l2)
    else:
        # https://en.wikipedia.org/wiki/Distance_from_a_point_to_a_line#Line_defined_by_two_points
        # 3.5 µs ± 29.9 ns per loop (mean ± std. dev. of 7 runs, 100,000 loops each)
        A, B = l1 - l2
        distance = -(
            A * point[1] - B * point[0] + l2[0] * l1[1] - l2[1] * l1[0]
        ) / sqrt(A ** 2 + B ** 2)

    return distance


def get_extreme_points(cnt):
    leftmost = cnt[cnt[:, 0] == cnt[cnt[:, 0].argmin()][0]].mean(axis=0)
    rightmost = cnt[cnt[:, 0] == cnt[cnt[:, 0].argmax()][0]].mean(axis=0)
    topmost = cnt[cnt[:, 1] == cnt[cnt[:, 1].argmin()][1]].mean(axis=0)
    bottommost = cnt[cnt[:, 1] == cnt[cnt[:, 1].argmax()][1]].mean(axis=0)
    # order l,t,r,b as in FCOS
    return leftmost, topmost, rightmost, bottommost


def get_eigen_points(points):
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


def ltrb(x, l, t, r, b):
    return x[0] - l, x[1] - t, r - x[0], b - x[1]
