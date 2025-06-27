#!/usr/bin/env python

import sys
import pylab
import numpy as np
import copy

# pylab.ion()

from show_annotations import (
    get_objects_of_interest,
    load_json,
    get_spherical_basis,
    get_chebyshev_basis,
)

fname = "soleil_proxima_dataset/autocenter_100161_Wed_Jan_27_12:21:02_2021_bright_failed.json"
json = load_json(fname)
oois = get_objects_of_interest(json)

look_at_me = "area_of_interest"
aoip = oois["properties"][oois["labels"].index(look_at_me)]

pylab.figure()
pylab.imshow(aoip.get_distance_transform())
ax1 = pylab.gca()
db = pylab.Polygon(aoip.get_dense_boundary(), color="red", fill=False, ls="solid")
ax1.add_patch(db)
cb = pylab.Polygon(
    aoip.get_boundary_from_chebyshev(domain=(-1, 1)),
    color="green",
    fill=False,
    ls="dotted",
)
ax1.add_patch(cb)


ltrb_bbox = aoip.get_ltrb_bbox()

fig, axes = pylab.subplots(2, 2)
for k, ax in enumerate(axes.flatten()):
    ax.imshow(ltrb_bbox[:, :, k])
    ax.set_title(f"{k}")


ltrb_boundary = aoip.get_ltrb_boundary()
fig, axes = pylab.subplots(2, 2)
for k, ax in enumerate(axes.flatten()):
    ax.imshow(ltrb_boundary[:, :, k])
    ax.set_title(f"{k}")

pylab.show()

sys.exit()

tap = aoip.get_thetas_and_points()
extend = -17
full_tap = copy.copy(tap)
if extend > 0:
    # hack to account for periodicity
    add_after_end = copy.copy(tap[:extend, :])  # [::-1, :]
    add_after_end[:, 0] = add_after_end[:, 0] + 2 * np.pi
    add_before_start = copy.copy(tap[-extend:, :])  # [::-1, :]
    add_before_start[:, 0] = add_before_start[:, 0] - 2 * np.pi
    tap = np.vstack((add_before_start, tap))
    tap = np.vstack((tap, add_after_end))
elif extend < 0:
    tap = tap[-extend:extend, :]

sph_degree = 7
sph_basis = get_spherical_basis(
    sph_degree, degree_step=1, order_step=2, order_max=7, points=full_tap[:, 0]
)
print(f"sph_basis.shape {sph_basis.shape}")
# mean_r = np.mean(full_tap[:, -1])
# mean_tap = copy.copy(full_tap)
# mean_tap[:,-1] -= mean_r
sph_coeff = aoip.get_sph_coeff(sph_degree, basis=sph_basis, tap=full_tap)
rsph = np.real(np.dot(sph_basis, sph_coeff))
print(f"rsph.shape {rsph.shape}")
sph_error = rsph.flatten() - full_tap[:, -1]
print(f"sph_error: {sph_error.shape}")
print(
    f"sph MAE: {np.sum(np.abs(sph_error))}, MSE: {np.sqrt(np.sum(np.power(sph_error, 2)))}"
)

ft = full_tap[:, 0]
fr = full_tap[:, -1]

che_degree = 6
factor = 1.2
try:
    for nviews in [8]:  # 3, 4, 6, 8]:
        view_range = 2 * np.pi / nviews
        viewis = []
        views = []
        view_centers = np.linspace(-np.pi, np.pi, nviews, endpoint=False)
        rches = []
        rvs = []
        ts = []

        for vc in view_centers:
            lefti = vc - (view_range) / 2
            righti = vc + (view_range) / 2
            left = vc - (factor * view_range) / 2
            right = vc + (factor * view_range) / 2
            viewi = view = np.logical_or(
                np.logical_and(lefti <= ft, ft < righti),
                np.logical_and(lefti + 2 * np.pi <= ft, ft < righti + 2 * np.pi),
            )
            view = np.logical_or(
                np.logical_and(left <= ft, ft < right),
                np.logical_and(left + 2 * np.pi <= ft, ft < right + 2 * np.pi),
            )

            viewis.append(viewi)
            views.append(view)

            tvi = ft[viewi]
            rvi = ft[viewi]
            tv = ft[view]
            rv = fr[view]

            print(f"vc, {vc}")
            try:
                print(f"len(tv) tv.min, tv.max: {len(tv)}, {tv.min()}, {tv.max()}")
            except:
                pass
            tvm = tv - vc
            tvm = (tvm + np.pi) % (2 * np.pi) - np.pi
            si = tvm.argsort()
            tvms = tvm[si]
            rvms = rv[si]

            tvim = tvi - vc
            tvim = (tvim + np.pi) % (2 * np.pi) - np.pi
            sii = tvim.argsort()
            tvims = tvim[sii]
            rvims = rvi[sii]

            try:
                print(
                    f"len(tvm) tvm.min, tvm.max: {len(tvm)}, {tvm.min()}, {tvm.max()}"
                )
            except:
                pass
            che_basis = get_chebyshev_basis(che_degree, points=tvm)

            x, residuals, rank, s = np.linalg.lstsq(che_basis, rvms, rcond=None)
            rche = np.array(np.dot(che_basis, x)).flatten()

            s = np.argwhere(tvms == tvims.min()).flatten()[0]
            e = np.argwhere(tvms == tvims.max()).flatten()[0]
            rvs.append(rvms[s : e + 1])
            ts.append((tvms[s : e + 1] + vc) % (2 * np.pi))
            rches.append(rche[s : e + 1])

        rv = np.array([])
        rc = np.array([])
        tv = np.array([])

        for r in rvs:
            rv = np.hstack([rv, r]) if len(rv) else r
        for t in ts:
            tv = np.hstack([tv, t]) if len(tv) else t
        for r in rches:
            rc = np.hstack([rc, r]) if len(rc) else r

        chestep_error = rc - rv

        print(
            f"nviews {nviews}: che step MAE: {np.sum(np.abs(chestep_error))}, MSE: {np.sqrt(np.sum(np.power(chestep_error, 2)))}"
        )
except:
    pass

t = full_tap[:, 0]
r = full_tap[:, -1]

che_degree = 19
# che_basis = get_chebyshev_basis(che_degree, points=t)
# print(f"che_basis.shape {che_basis.shape}")
# che_coeff = aoip.get_chebyshev(degree=che_degree)
# rche = np.dot(che_basis, che_coeff)
# che_error = rche - r
## print(f"che_error {che_error}")
# print(
# f"che MAE: {np.sum(np.abs(che_error))}, MSE: {np.sqrt(np.sum(np.power(che_error, 2)))}"
# )

tp = np.hstack([t, t + 2 * np.pi])
rp = np.hstack([r, r])

tp = np.hstack([t - 2 * np.pi, tp])
rp = np.hstack([r, rp])

from scipy.interpolate import CubicSpline, interp1d

# interp = CubicSpline(tp, rp, bc_type="periodic")
interp = interp1d(tp, rp)

tinterp = np.pi * np.linspace(-1.01, 1.01, 401)
rinterp = interp(tinterp)
# s = len(t) - 0
# e= 2*len(t) + 0
# si = np.argsort(tp)
# tp = tp[si]
# rp = rp[si]
# tinterp = tp[s: e]
# rinterp = rp[s: e]
cheinterp_basis = get_chebyshev_basis(che_degree, points=tinterp)
print(f"cheinterp_basis.shape {cheinterp_basis.shape}")
# x, residuals, rank, s = np.linalg.lstsq(cheinterp_basis, rinterp, rcond=None)
x = np.polynomial.chebyshev.chebfit(tinterp, rinterp, che_degree)
rcheinterp = np.array(np.dot(cheinterp_basis, x)).flatten()
si = np.logical_and(tinterp > -np.pi, tinterp < np.pi)
tinterpv = tinterp[si]
rcheinterpv = rcheinterp[si]
rinterpv = rinterp[si]

cheinterp_error = rcheinterpv - rinterpv
print(
    f"che interp MAE: {np.sum(np.abs(cheinterp_error))}, MSE: {np.sqrt(np.sum(np.power(cheinterp_error, 2)))}"
)

# sitv = np.argsort(tv)
# tvm = tv[sitv]
# rcm = rc[sitv]

pylab.figure()
pylab.plot(full_tap[:, 0], full_tap[:, -1], label="manual")
pylab.plot(tinterp, rinterp, label="interp")

# pylab.plot(full_tap[:, 0]% (2*np.pi), np.array(rsph).flatten(), label="spherical")

# for k, (ts, rs) in enumerate(zip(ts, rches)):
# pylab.plot(ts, np.array(rs).flatten(), "o", label=f"stepped chebyshev {k}")
# pylab.plot(tvm, rcm, label=f"stepped chebyshev {len(views)}")
# pylab.plot(t, np.array(rche).flatten(), "-.", label="chebyshev")
pylab.plot(tinterpv, rcheinterpv, "--", label="interp cheybshev")
pylab.legend()

# pylab.figure()
# pylab.plot(np.real(sph_coeff), label="spherical")
# pylab.plot(che_coeff, label="chebyshev")
# pylab.legend()

pylab.show()
