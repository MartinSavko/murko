#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import sys
import time
import random
import traceback
import numpy as np
import cv2 as cv
import skimage as ski
import pylab
import seaborn as sns

from objects_of_interest import (
    get_objects_of_interest,
    update_maps,
    merge_maps,
    get_label_points,
    adjust_points,
)

from regionprops import (
    Regionprops,
    get_mask_from_polygon,
    get_offsets,
    get_centerness,
    get_distance_transform,
    get_universal_ltrb,
)

from config import (
    notion_importance,
    keypoints,
    keypoint_labels,
    global_keypoints,
    named_points_colors,
    classifications,
)

from keypoints import (
    get_origin,
    get_most_likely_click,
    get_extreme,
    get_start_possible,
    get_start_likely,
    get_ltrbc,
    get_orientation_and_direction,
    get_named_pca_points,
    get_gang_of_five,
    # draw_point,
)

from utils import get_valid_image_and_points

def timeit(func):
    # https://stackoverflow.com/questions/1622943/timeit-versus-timing-decorator
    def timed(*args, **kw):
        ts = time.time()
        result = func(*args, **kw)
        te = time.time()

        print("func:%r took: %2.8f sec" % (func.__name__, te - ts))
        return result

    return timed


def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x


def get_flipped_image(image, axis):
    flipped_image = flip_axis(image, axis)
    return flipped_image


def get_flipped_img_and_points(img, points):
    axis = random.choice([0, 1])
    fimg = get_flipped_image(img, axis)
    fpoints = points[:, :]
    fpoints[:, axis] = img.shape[axis] - points[:, axis]
    return fimg, fpoints


def get_transposed_image(image):
    new_axes_order = (1, 0) + tuple(range(2, len(image.shape)))
    transposed_image = np.transpose(image, new_axes_order)
    return transposed_image


def get_transposed_img_and_points(img, points):
    timg = get_transposed_image(img)
    tpoints = points[:, ::-1]
    return timg, tpoints


#@timeit
def make_points_homogeneous(points):
    hpoints = np.append(points, np.ones((points.shape[0], 1)), axis=1)
    return hpoints


def get_transformed_points(points, transformation_matrix, order=[1, 0, 2]):
    if len(points.shape) == 2:
        points = make_points_homogeneous(points)
    points = points[:, order]
    transformed_points = np.dot(transformation_matrix, points.T).T
    transformed_points = transformed_points[:, order]
    return transformed_points[:, :2]


def get_output_shape(input_shape, transformation_matrix):
    corners = get_corners()
    print("corners")
    print(corners)
    print(f"input_shape {input_shape}")
    print(f"transformation_matrix")
    print(transformation_matrix)
    corners *= input_shape[::-1]

    print(f"corners {corners}")
    hcorners = make_points_homogeneous(corners)
    print(f"hcorners {hcorners}")
    tcorners = get_transformed_points(hcorners, transformation_matrix)
    print(f"tcorners {tcorners}")
    # distances = np.abs(tcorners[:-1, :2] - tcorners[-1, :2])
    # print(f'distances {distances}')
    # output_shape = np.max(tcorners[:, :2], axis=0)
    output_shape = np.max(tconrners[:, :2], axis=0) - np.min(tcorners[:, :2], axis=0)
    print(f"output_shape {output_shape}")
    return output_shape.astype(int)


#@timeit
def estimate_transformation(src, dst):
    # https://docs.opencv.org/3.4.8/d4/d61/tutorial_warp_affine.html
    srcTri = src[:3, ::-1].astype("float32")
    dstTri = dst[:3, ::-1].astype("float32")
    transformation = cv.getAffineTransform(srcTri, dstTri)
    return transformation


#@timeit
def get_transformed_img_and_points(
    img,
    points,
    transformation=None,
    valid=True,
    verbose=False,
    return_optional=False,
    preserve_shape=True,
):

    if transformation is None:
        transformation = get_random_transformation(img_shape=np.array(img.shape[:2]))

    tpoints = get_transformed_points(points, transformation._inv_matrix)
    corners = get_corners() * np.array(img.shape[:2])
    tcorners = get_transformed_points(corners, transformation._inv_matrix).astype(
        "int32"
    )

    tc_min = tcorners.min(axis=0)
    tc_max = tcorners.max(axis=0)

    if verbose:
        print("tcorners", tcorners, tcorners.dtype)
        print(f"tc_min {tc_min} tc_max {tc_max}")
        print(f"tpoints.min(axis=0) {tpoints.min(axis=0)}")

    tpoints = tpoints - tc_min
    tcorners = tcorners - tc_min
    tc_min = tcorners.min(axis=0)
    tc_max = tcorners.max(axis=0)
    output_shape = np.floor(tc_max - tc_min)[:2]
    # output_shape -= np.array([10, 10])
    # output_shape = cv.boundingRect(tcorners)[2:]
    warp_mat = estimate_transformation(corners, tcorners)
    timage = cv.warpAffine(
        img,
        warp_mat,
        output_shape, #[:2][::-1],
    )
    # timage = get_transformed_image(
    #     img, transformation, output_shape=output_shape2
    # )
    if verbose:
        print("after adjustment tcorners", tcorners, tcorners.dtype)
        print(f"after tc_min {tc_min} tc_max {tc_max}")
        print(f"after tpoints.min(axis=0) {tpoints.min(axis=0)}")
        output_shape2 = cv.boundingRect(tcorners)[2:]
        print(f"output_shape {output_shape} from boundingRect {output_shape2}")
        print(f"warp_mat\n{warp_mat}")

    otimage = timage.copy()
    valid_shift = np.array([0, 0])

    if valid:
        timage, tpoints, valid_shift = get_valid_image_and_points(tcorners, timage, tpoints, verbose=verbose)

    if preserve_shape and size_differs(img.shape[:2], timage.shape[:2]):
        timage, tpoints = resize_img_and_points(
            timage, tpoints, img.shape[:2], fractional=False, verbose=verbose
        )

    return_value = timage, tpoints
    if return_optional:
        return_value += otimage, valid_shift
    return return_value


def get_shifted_image(image, tx=None, ty=None, max_shift=0.25, valid=True):
    # https://www.kaggle.com/code/ahmedabdelfattah20/image-augmentation-using-opencv
    rows, cols, chans = image.shape
    if tx is None:
        tx = random.randint(int(-max_shift * cols), int(max_shift * cols))
    if ty is None:
        ty = random.randint(int(-max_shift * rows), int(max_shift * rows))
    M = np.float32([[1, 0, tx], [0, 1, ty]])
    shifted_image = cv.warpAffine(image, M, (cols, rows))
    x, y = 0, 0
    if valid:
        x, y = max(tx, 0), max(ty, 0)
        w, h = cols - abs(tx), rows - abs(ty)
        shifted_image = shifted_image[y : y + h, x : x + w]
    return shifted_image, x, y


def get_corners():
    # corners = np.array([[0, 0], [1, 0], [0, 1], [1, 1]])
    corners = np.array([[0, 0], [1, 0], [1, 1], [0, 1]])
    return corners


#@timeit
def get_shifted_img_and_points(
    img, points, max_shift=0.33, valid=True, preserve_shape=True, verbose=False
):
    rows, cols, chans = img.shape
    tx = random.randint(int(-max_shift * cols), int(max_shift * cols))
    ty = random.randint(int(-max_shift * rows), int(max_shift * rows))
    if verbose:
        print(
            f"points min {np.round(points.min(axis=0), 1)}, max {np.round(points.max(axis=0), 1)}"
        )

    shift = np.array([ty, tx])
    shifted_img, xo, yo = get_shifted_image(img, tx, ty, valid=valid)
    points = points + shift

    if verbose:
        print(f"shift: ty {ty} tx {tx}")

    if preserve_shape and size_differs((rows, cols), shifted_img.shape[:2]):
        if valid:
            valid_shift = np.array([yo, xo])
            points = points - valid_shift

        if verbose:
            print(f"preserving shape from {shifted_img.shape[:2]} to {(rows, cols)}")
            print(f"valid_shift: yo {yo} xo {xo}")
            print(f"shift: ty {ty} tx {tx}")
            print(f"shift - valid_shift = {shift - valid_shift}")

            print(
                f"points min {np.round(points.min(axis=0), 1)}, max {np.round(points.max(axis=0), 1)}"
            )
            print(f"shapes: original {(rows, cols)} shifted {shifted_img.shape[:2]}")

        shifted_img, points = resize_img_and_points(
            shifted_img, points, (rows, cols), fractional=False, verbose=verbose
        )

    if verbose:
        print(f"shifted_image.shape {shifted_img.shape[:2]}")
        print(
            f"points min {np.round(points.min(axis=0), 1)}, max {np.round(points.max(axis=0), 1)}"
        )

    return shifted_img, points


def resize_img_and_points(img, points, required_shape, fractional=False, verbose=False):
    input_shape = np.array(img.shape[:2])
    output_shape = np.array(required_shape)
    resize_factor = output_shape / input_shape
    if verbose:
        print(f"resize_factor {resize_factor}")
    img = get_resized_image(img, required_shape, anti_aliasing=True)

    if not fractional:
        points = points * resize_factor
    return img, points


def get_noisy_image(image, hmax=100, smax=20, vmax=10, verbose=False):
    # https://www.kaggle.com/code/ahmedabdelfattah20/image-augmentation-using-opencv
    rows, cols, chans = image.shape
    if verbose:
        print(f"adding noise hmax, smax, vmax {hmax} {smax} {vmax}")
        print(f"image.dtype {image.dtype}")
    image = check_uint8(image)
    hsv = cv.cvtColor(image, cv.COLOR_RGB2HSV)
    h, s, v = cv.split(hsv)
    h += np.random.randint(0, hmax, size=(rows, cols), dtype=np.uint8)
    s += np.random.randint(0, smax, size=(rows, cols), dtype=np.uint8)
    v += np.random.randint(0, vmax, size=(rows, cols), dtype=np.uint8)

    noisy_hsv = cv.merge([h, s, v])
    noisy_image = cv.cvtColor(noisy_hsv, cv.COLOR_HSV2RGB) / 255.0
    return noisy_image


def get_blurred_image(image, mini=2, maxi=7, verbose=False):
    # https://www.kaggle.com/code/ahmedabdelfattah20/image-augmentation-using-opencv
    blur_val = random.randint(mini, maxi)  # blur value random
    if verbose:
        print(f"blur parameters {blur_val}")
    blurred_image = cv.blur(image, (blur_val, blur_val))
    return blurred_image


def check_uint8(img):
    if (
        (img.dtype != "uint8" and img.max() <= 1)
        or img.dtype == np.float32
        or img.dtype == np.float64
    ):
        img = (img * 255).astype("uint8")
    return img


def get_gray(img, doer="cv"):
    if doer == "cv":
        img = check_uint8(img)
        gray = cv.cvtColor(img, cv.COLOR_RGB2GRAY)
        gray = cv.cvtColor(gray, cv.COLOR_GRAY2RGB) / 255.0
    elif doer == "np":
        gray = img.mean(axis=2)
        gray = np.stack([gray] * 3, axis=2)
    return gray


def get_gamma_image(image, gamma=None, gamma_min=0.2, gamma_max=5.0, verbose=False):
    # https://docs.opencv.org/4.13.0/d3/dc1/tutorial_basic_linear_transform.html
    image = check_uint8(image)
    if gamma is None:
        gamma = gamma_min + random.random() * (gamma_max - gamma_min)
    if verbose:
        print(f"gamma {gamma:.3f}")
    lookUpTable = np.empty((1, 256), np.uint8)
    for i in range(256):
        lookUpTable[0, i] = np.clip(pow(i / 255.0, gamma) * 255.0, 0, 255)
    try:
        gamma_image = cv.LUT(image, lookUpTable) / 255.0
    except:
        traceback.print_exc()
        print("image.shape", image.shape)
        print("image.dtype", image.dtype)
        gamma_image = image / 255.0
    return gamma_image


#@timeit
def get_transformed_image(
    img, transformation, output_shape=None, doer="cv", cval=0, mode="constant"
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
            output_shape[:2][::-1],
            borderValue=cval,
            borderMode=borderMode,
        )
    print(f"tranformed image shape {transformed_image.shape}, input_shape {img.shape}")
    return transformed_image


def get_resized_image(
    img,
    img_size,
    anti_aliasing=True,
    interpolation="INTER_AREA",
    doer="cv",
    smart_interpolation=True,
):
    if doer == "ski":
        resized_image = ski.transform.resize(img, img_size, anti_aliasing=anti_aliasing)
    elif doer == "cv":
        # https://opencv.org/blog/resizing-and-rescaling-images-with-opencv/
        # Method	        Description	Best               Used For
        # INTER_NEAREST	Nearest-neighbor interpolation (fastest, but low quality)
        #                                               Simple, fast resizing (e.g.,
        #                                               pixel art, binary images)
        # INTER_LINEAR	Bilinear interpolation        	General-purpose
        #                                               resizing (good balance of speed
        #                                               & quality)
        # INTER_CUBIC	Bicubic interpolation           High-quality upscaling,
        #               (uses 4×4 pixel neighborhood)   smoother results
        # INTER_AREA	    Resampling                      Best for shrinking images
        #               using pixel area relation       (avoids aliasing)
        # INTER_LANCZOS4	Lanczos interpolation           High-quality upscaling &
        #               using 8×8 pixel neighborhood    downscaling (preserves fine
        #                                               details)
        if smart_interpolation and np.prod(img_size) > np.prod(img.shape[:2]):
            interpolation = "INTER_LINEAR"
        resized_image = cv.resize(
            img, img_size[::-1], interpolation=getattr(cv, interpolation)
        )
    return resized_image


# zoom_factor=0.25,
# shift_factor=0.25,
# shear_factor=45,
# default_transform_gang=[0, 0, 0, 0, 1, 1],


#@timeit
def get_random_transformation(
    rotation_range=np.pi / 4,
    scale_range=0.5,
    shear_range=np.pi / 6,
    img_shape=np.array((1200, 1600)),
    rotation_center="center",
    verbose=False,
    # translation_range=0.25,
):
    if rotation_center == "random":
        r_center = np.random.random(size=2) * img_shape
    else:
        r_center = np.array(img_shape) / 2

    shift_c = ski.transform.AffineTransform(translation=-r_center)
    shift_invc = ski.transform.AffineTransform(translation=+r_center)

    rotation = (np.random.rand() - 0.5) * rotation_range
    scale = 1 + (np.random.random(size=2) - 0.5) * scale_range
    shear = (np.random.random(size=2) - 0.5) * shear_range
    # translation = [
    #     0,
    #     0,
    # ]  # (np.random.random(size=2) - 0.5) * translation_range * img_shape
    if verbose:
        print(f"rotation {rotation}, rotation_center {r_center}")
        print(f"scale {scale}")
        print(f"shear {shear}")
        # print(f"translation {translation}")

    t_rotation = ski.transform.AffineTransform(rotation=rotation)
    t_scale = ski.transform.AffineTransform(scale=scale)
    t_shear = ski.transform.AffineTransform(shear=shear)
    # t_translation = ski.transform.AffineTransform(translation=translation)
    #
    # random_transformation = ski.transform.AffineTransform(
    # scale=scale, rotation=rotation, shear=shear, translation=translation
    # )

    # random_transformation = (
    #     t_rotation + t_scale + t_shear
    # )

    random_transformation = shift_c + t_rotation + shift_invc + t_scale + t_shear
    return random_transformation


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


def size_differs(original_size, img_size):
    return original_size[0] != img_size[0] or original_size[1] != img_size[1]


def _get_unmasked_image(image, mask):
    image[mask.astype(bool) == False] = 0
    return image


def get_unmasked_image(image, masks, label):
    # image[masks[label].astype(bool) == False] = 0
    image = _get_unmasked_image(image, masks[label])
    return image


def swap_backgrounds(img, foreground_mask, new_background, img_shape=None):
    if img_shape is None:
        img_shape = img.shape[:2]

    if size_differs(img_shape, new_background.shape[:2]):
        new_background = get_resized_image(
            new_background, img_shape, anti_aliasing=True
        )
    img[foreground_mask == 0] = new_background[foreground_mask == 0]
    return img


def get_augment_control(
    threshold=0.5,
    transform=True,
    shift=True,
    transpose=True,
    flip=True,
    swap_backgrounds=True,
    black_and_white=True,
    random_brightness=True,
    random_contrast=True,
    random_channel_shift=True,
    random_gamma=False,
    random_blur=False,
    random_noise=False,
    verbose=False,
):
    do_flip = False
    do_transpose = False
    do_transform = False
    do_shift = False
    do_swap_backgrounds = False
    do_black_and_white = False
    do_random_brightness = False
    do_random_contrast = False
    do_random_channel_shift = False
    do_random_gamma = False
    do_random_blur = False
    do_random_noise = False

    if flip and random.random() < 0.75:
        do_flip = True

    if transpose and random.random() < threshold:
        do_transpose = True
        if flip and random.random() < threshold:
            do_flip = True
    else:
        if flip and random.random() < threshold:
            do_flip = True

    if transform and random.random() < threshold:
        do_transform = True
    # do_transform = True

    if shift and random.random() < threshold:
        do_shift = True

    if swap_backgrounds and random.random() < threshold / 2:
        do_swap_backgrounds = True
    # do_swap_backgrounds = True
    if black_and_white and random.random() < threshold / 2:
        do_black_and_white = True

    if random_brightness and random.random() < threshold / 2:
        do_random_brightness = True

    if random_contrast and random.random() < threshold / 2:
        do_random_contrast = True

    if (
        random_channel_shift
        # # and not do_black_and_white
        and random.random() < threshold / 2
    ):
        do_random_channel_shift = True

    if random_gamma and random.random() < threshold / 2:
        do_random_gamma = True
    if random_blur and random.random() < threshold / 2:
        do_random_blur = True
    if random_noise and random.random() < threshold / 2:
        do_random_noise = True

    if verbose:
        print(f"do_transpose: {do_transpose}")
        print(f"do_flip: {do_flip}")
        print(f"do_transform: {do_transform}")
        print(f"do_shift: {do_shift}")
        print(f"do_swap_backgrounds: {do_swap_backgrounds}")
        print(f"do_black_and_white: {do_black_and_white}")
        print(f"do_random_brightness: {do_random_brightness}")
        print(f"do_random_contrast: {do_random_contrast}")
        print(f"do_random_channel_shift: {do_random_channel_shift}")
        print(f"do_random_gamma: {do_random_gamma}")
        print(f"do_random_blur: {do_random_blur}")
        print(f"do_random_noise: {do_random_noise}")
    return (
        do_flip,
        do_transpose,
        do_transform,
        do_shift,
        do_swap_backgrounds,
        do_black_and_white,
        do_random_brightness,
        do_random_contrast,
        do_random_channel_shift,
        do_random_gamma,
        do_random_blur,
        do_random_noise,
    )


def draw_point(point, ax=None, radius=3, color="red"):
    if ax is None:
        ax = pylab.gca()

    p = pylab.Circle(point, radius=radius, color=color)
    ax.add_patch(p)

class Sample:
    def __init__(
        self,
        json_file,
        notion_importance=notion_importance,
        preferred_image_size=None,
        not_to_keep=["masks"],
    ):
        self.oois = get_objects_of_interest(json_file)
        self.realpath = self.oois["realpath"]
        self.json_path = self.oois["json_path"]
        self.image_path = self.oois["image_path"]
        self.image_shape = self.oois["image_shape"]
        self.points = self.oois["points"]
        self.indices = self.oois["indices"]
        self.labels = self.oois["labels"]
        self.fractional = self.oois["fractional"]
        self.notion_importance = notion_importance
        self.preferred_image_size = preferred_image_size

        if self.preferred_image_size is not None and size_differs(
            self.preferred_image_size, self.image_shape
        ):
            self.image_at_preferred_resolution, points = (
                resize_img_and_points(
                    self.get_image(preferred=False),
                    self.get_points(preferred=False),
                    self.preferred_image_size,
                    fractional=self.fractional,
                )
            )

            # if "autocenter_100161_Thu_Jan__6_14:25:41_2022_bright_failed.json" in self.json_path:
            #     print("points before\n", points)

            points = adjust_points(points, self.preferred_image_size)

            # if "autocenter_100161_Thu_Jan__6_14:25:41_2022_bright_failed.json" in self.json_path:
            #     print("points after\n", points)
            self.points_at_preferred_resolution = points

        else:
            self.image_at_preferred_resolution = self.image
            self.points_at_preferred_resolution = self.points

        for key in not_to_keep:
            if key in self.oois:
                del self.oois[key]

    def get_target(self, head, img, points):
        pass

    def get_blank_hierarchy(self, notions, image_shape=None):
        if image_shape is None:
            image_shape = self.get_image_shape()

        blank_hierarchy = np.zeros(image_shape + (len(notions),), dtype=np.int8)

        return blank_hierarchy

    def get_image(self, preferred=False):
        if preferred:
            return self.image_at_preferred_resolution
        return self.oois["image"].copy()

    def get_image_path(self):
        return self.image_path

    def get_image_shape(self):
        return tuple(map(int, self.image_shape))

    def get_points(self, preferred=False):
        if preferred:
            return self.points_at_preferred_resolution
        return self.points.copy()

    def get_indices(self):
        return self.indices

    def get_labels(self):
        return self.labels

    def get_label_points(self, label, points, image_shape):
        label_points = []
        # https://stackoverflow.com/questions/6294179/how-to-find-all-occurrences-of-an-element-in-a-list
        indices = [i for i, x in enumerate(self.labels) if x == label]
        # idx = self.labels.index(label) if label in self.labels else None
        for idx in indices:
            i_start, i_end = self.indices[idx]
            lps = points[i_start: i_end]
            if self.fractional:
                lps *= image_shape
            label_points.append(lps)

        return label_points

    def _get_properties(self, points=None, image_shape=None, exclusive_label=None):
        if points is None:
            points = self.get_points()
        if image_shape is None:
            image_shape = self.get_image_shape()

        properties = []
        for k, label in enumerate(self.labels):
            i_start, i_end = self.indices[k]
            ps = points[i_start:i_end]
            if self.fractional:
                ps *= image_shape
            negative_points = []
            if label in [
                "plastic",
                "aether",
                "area_of_interest_aether",
                "crystal_aether",
            ]:
                if label == "plastic":
                    complement = "loop_inside"
                elif label == "aether":
                    complement = "foreground"
                else:
                    complement = label.replace("_aether", "")

                negative_points = self.get_label_points(complement, points, image_shape)

            props = Regionprops(
                ps,
                image_shape=image_shape,
                distance_transform_pad=0 if "aether" in label else 2,
                negative_points=negative_points,
            )
            properties.append(props)

        return properties

    def _get_maps(
        self,
        points=None,
        image_shape=None,
        properties=None,
        kind="mask",
        method="logical_or",
        normalize=True,
        exclusive_label=None,
        **kwargs,
    ):
        if properties is None:
            properties = self._get_properties(points, image_shape)

        _maps = {}
        for k, label in enumerate(self.labels):
            if exclusive_label:
                if label != exclusive_label:
                    continue
            _map = getattr(properties[k], f"get_{kind}")(**kwargs)
            if len(_map.shape) == 2:
                update_maps(_maps, label, _map, method=method)
            else:
                print(f"possible problem {_map.shape} shape is wrong, please check")

        if "mask" not in kind and normalize:
            for _n, _m in _maps.items():
                _min = _m.min()
                _max = _m.max()
                if _min != _max:
                    _maps[_n] = (_m - _min) / (_max - _min)

        if exclusive_label:
            _maps = _maps[exclusive_label]

        return _maps

    def get_masks(self, points=None, image_shape=None):
        masks = self._get_maps(points, image_shape, kind="mask", method="logical_or")
        return masks

    def get_binary_segment(self, points=None, image_shape=None):
        return self.get_masks(points, image_shape)

    def get_hierarchy(
        self,
        points=None,
        image_shape=None,
        masks=None,
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

        notions.sort(key=lambda x: -self.notion_importance[x])
        values = dict((notion, k) for k, notion in enumerate(notions))

        hierarchy = self.get_blank_hierarchy(notions, image_shape=image_shape)

        if masks is None:
            masks = self.get_masks(points=points, image_shape=image_shape)

        for notion in notions:
            if notion in masks:
                hierarchy[:, :, values[notion]] = (
                    masks[notion].astype(np.uint8) * values[notion]
                )

        return hierarchy

    def get_flat_hierarchy(
        self,
        points=None,
        image_shape=None,
        masks=None,
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
        hierarchy = self.get_hierarchy(
            points=points, image_shape=image_shape, masks=masks, notions=notions
        )
        flat_hierarchy = np.argmax(hierarchy, axis=2)
        return flat_hierarchy

    def get_categorical_hierarchy(
        self,
        points=None,
        image_shape=None,
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
        from tensorflow import keras

        flat_hierarchy = self.get_flat_hierarchy(points=points, notions=notions)
        categorical_hierarchy = keras.utils.to_categorical(
            flat_hierarchy, num_classes=len(notions)
        )

        return categorical_hierarchy

    def get_support_type(self):
        return self.oois["support_type"]

    def get_crystal_present(self):
        return "crystal" in self.labels

    def get_anything_present(self):
        return "foreground" in self.labels

    def get_precipitate_present(self):
        return "precipitate" in self.labels

    def get_ice_present(self):
        return "ice" in self.labels

    def get_global_classification_target(
        self,
        what="support_type",
        points=None,
        image_shape=None,
        masks=None,
        notions=None,
        focus="background",
    ):

        if notions is None:
            notions = classifications[what]

        classification_target = self.get_blank_hierarchy(
            notions, image_shape=image_shape
        )

        if masks is None:
            masks = self.get_masks(points=points, image_shape=image_shape)

        idx = notions.index(getattr(self, f"get_{what}")())
        classification_target[:, :, idx] = masks[focus].astype(np.uint8)

        return classification_target

    def get_distance_transform(self, points=None, image_shape=None):
        distance_transform = self._get_maps(
            points, image_shape, kind="distance_transform", method="max"
        )
        return distance_transform

    def get_inverse_distance_transform(self, points=None, image_shape=None):
        distance_transform = self._get_maps(
            points, image_shape, kind="inverse_distance_transform", method="max"
        )
        return distance_transform

    def get_sqrt_distance_transform(self, points=None, image_shape=None):
        distance_transform = self._get_maps(
            points, image_shape, kind="sqrt_distance_transform", method="min"
        )
        return distance_transform

    def get_sqrt_inverse_distance_transform(self, points=None, image_shape=None):
        distance_transform = self._get_maps(
            points, image_shape, kind="sqrt_inverse_distance_transform", method="max"
        )
        return distance_transform

    def get_power_distance_transform(self, points=None, image_shape=None):
        distance_transform = self._get_maps(
            points, image_shape, kind="power_distance_transform", method="min"
        )
        return distance_transform

    def get_power_inverse_distance_transform(self, points=None, image_shape=None):
        distance_transform = self._get_maps(
            points, image_shape, kind="power_inverse_distance_transform", method="max"
        )
        return distance_transform

    def get_centerness(self, points=None, image_shape=None):
        centerness = self._get_maps(
            points, image_shape, kind="centerness", method="max"
        )
        return centerness

    def get_offsets(self, point, image_shape=None):
        if image_shape is None:
            image_shape = self.get_image_shape()
        offset_h, offset_v = get_offsets(point, image_shape)
        return offset_h, offset_v

    def get_keypoint_centerness(self, keypoints, image_shape=None):
        if image_shape is None:
            image_shape = self.get_image_shape()
        keypoint_centerness = np.zeros(image_shape)
        for point in keypoints:
            _centerness = get_centerness(point, image_shape)
            keypoint_centerness = merge_maps(
                keypoint_centerness, _centerness, method="max"
            )
        return keypoint_centerness

    def get_keypoint_offsets(self, keypoints, image_shape=None):
        if image_shape is None:
            image_shape = self.get_image_shape()
        offsets_h, offsets_v = np.zeros(image_shape), np.zeros(image_shape)
        for point in keypoints:
            _h, _v = get_offsets(point, image_shape)
            offsets_h = merge_maps(offsets_h, _h, method="min")
            offsets_v = merge_maps(offsets_v, _v, method="min")
        return offsets_h, offsets_v

    def get_bbox_masks(self, points=None, image_shape=None):
        bbox_mask = self._get_maps(
            points, image_shape, kind="bbox_mask", method="logical_or"
        )
        return bbox_mask

    def get_bbox_ltrbs(self, points=None, image_shape=None):
        bbox_ltrb = self._get_maps(
            points, image_shape, kind="bbox_ltrb", method="logical_or"
        )
        return bbox_ltrb

    def get_ellipse_masks(self, points=None, image_shape=None):
        ellipse_mask = self._get_maps(
            points, image_shape, kind="ellipse_mask", method="logical_or"
        )
        return ellipse_mask

    def get_min_rectangle_masks(self, points=None, image_shape=None):
        min_rectangle_mask = self._get_maps(
            points, image_shape, kind="min_rectangle_mask", method="logical_or"
        )
        return min_rectangle_mask

    def _get_origin(self, labels, indices, points, properties):
        return get_origin(labels, indices, points, properties)

    def get_most_likely_click(self, labels, indices, points, properties, extreme=None):
        mlc = get_most_likely_click(
            labels, indices, points, properties, extreme=extreme
        )
        return mlc

    def get_origin_extreme(self, labels, indices, points, properties):
        o, e = get_origin_extreme(labels, indices, points, properties)
        return o, e

    def _get_extreme(self, labels, indices, points, properties):
        return get_extreme(labels, indices, points, properties)

    def _get_start_possible(self, labels, indices, points, properties):
        return get_start_possible(labels, indices, points, properties)

    def _get_start_likely(self, labels, indices, points, properties):
        return get_start_likely(labels, indices, points, properties)

    def _get_ltrbc(self, labels, indices, points, properties, label):
        return get_ltrbc(labels, indices, points, properties, label)

    def _get_named_pca_points(self, origin, projection, origin_is_extreme=False):
        return get_named_pca_points(
            origin, projection, origin_is_extreme=origin_is_extreme
        )

    def _guess_point(self, point_name):
        pns = point_name.split("_")
        label, kind = None, None
        if len(pns) == 2:
            abbreviation = pns[0]
            if pns[0] == "aoi":
                label = "area_of_interest"
            else:
                label = pns[0]
            kind = pns[1]
        return label, abbreviation, kind

    def _check_lipp(self, labels=None, indices=None, points=None, properties=None):
        l = labels if labels is not None else self.get_labels()
        i = indices if indices is not None else self.get_indices()
        p = points if points is not None else self.get_points()
        _p = properties if properties is not None else self._get_properties(points=p)
        return l, i, p, _p

    def get_gang_of_five(self, labels=None, indices=None, points=None, properties=None):
        args = self._check_lipp(labels, indices, points, properties)
        return get_gang_of_five(*args)

    def get_keypoints(
        self, named_points, labels=None, indices=None, points=None, properties=None
    ):
        args = self._check_lipp(labels, indices, points, properties)
        keypoints = {}
        ltrbcs = {}
        for point_name in named_points:
            if hasattr(self, f"_get_{point_name}"):
                keypoints[point_name] = getattr(self, f"_get_{point_name}")(*args)
            else:
                label, abbreviation, kind = self._guess_point(point_name)
                if label in args[0] and label not in ltrbcs:
                    # ltrbcs[label] = self._get_ltrbc(*args + (label,))
                    projection = args[-1][args[0].index(label)].get_mask()
                    if label == "pin":
                        origin_is_extreme = True
                        origin = self._get_start_possible(*args)
                    else:
                        origin_is_extreme = False
                        origin = self._get_origin(*args)
                    ltrbcs[label] = self._get_named_pca_points(
                        origin, projection, origin_is_extreme=origin_is_extreme
                    )
                    print(f"ltrbcs {ltrbcs}")
                print(label, abbreviation, kind)
                if label in ltrbcs:
                    keypoints[point_name] = ltrbcs[label][kind]

        return keypoints

    # def draw_voronoi(img, subdiv) :
    # ...:
    # ...:     ( facets, centers) = subdiv.getVoronoiFacetList([])
    # ...:     for f, c in zip(facets, centers) :
    # ...:         f = f.astype(np.int32)
    # ...:         c = c.astype(np.int16)
    # ...:         color = random.randint(0, 255)
    # ...:         cv2.fillConvexPoly(img, f, color);
    # ...:         cv2.polylines(img, [f], True, (0, 0, 0), 1)
    # ...:         cv2.circle(img, c, 33, (0, 0, 0))

    def get_aoi_keypoints(self, origin=None, label="area_of_interest"):
        lipp = self._check_lipp()
        if origin is None:
            origin = get_origin(*lipp)

        npp = {
            "origin": origin,
            "left": np.array((-1, -1)),
            "right": np.array((-1, -1)),
            "top": np.array((-1, -1)),
            "bottom": np.array((-1, -1)),
        }

        labels = lipp[0]
        if label in labels:
            projection = lipp[-1][lipp[0].index(label)].get_mask()
            npp.update(get_named_pca_points(origin, projection))

        return npp

    def get_voronoi(
        self,
        keypoints,  # most_likely_click, aoi_start, aoi_end, aoi_top, aoi_bottom, start_possible, origin
        image_shape=None,
        verbose=False,
    ):
        if image_shape is None:
            image_shape = self.get_image_shape()

        """http://learnopencv.com/delaunay-triangulation-and-voronoi-diagram-using-opencv-c-python/"""

        voronoi = np.zeros(image_shape, dtype=np.int8)
        subdiv = cv.Subdiv2D((0, 0, image_shape[1], image_shape[0]))

        present_points = []
        for key, point in keypoints.items():
            if point is not None and -1 not in point:
                print(f"{key}, {point}")
                pt = point.astype(np.int16)
                subdiv.insert(pt)
                present_points.append(key)
        if verbose:
            print(f"present_points {present_points}")
        facets, centers = subdiv.getVoronoiFacetList([])
        if verbose:
            print(f"facets {facets}")
            print(f"centers {centers}")
            print(f"len(facets) {len(facets)}")
            print(f"len(centers) {len(centers)}")
            print(f"len(present_points) {len(present_points)}")
        for i, key in enumerate(present_points):
            label = keypoint_labels[key]
            if i < len(facets):
                ifacets = facets[i].astype(np.int32)
                cv.fillConvexPoly(voronoi, ifacets, label)

        # voronoi = voronoi[:image_shape[0], :image_shape[1]]
        return voronoi

    def get_image_and_points(
        self,
        img_size=None,
        augment=False,
        do_transform=False,
        new_background=None,
        require_transpose=False,
        disallow_transpose=False,
        verbose=False,
        preferred=False,
    ):

        img = self.get_image(preferred=preferred)
        points = self.get_points(preferred=preferred)

        (
            do_flip,
            do_transpose,
            do_transform,
            do_shift,
            do_swap_backgrounds,
            do_black_and_white,
            do_random_brightness,
            do_random_contrast,
            do_random_channel_shift,
            do_random_gamma,
            do_random_blur,
            do_random_noise,
        ) = get_augment_control(verbose=verbose, transform=do_transform)

        already_transposed = False
        if require_transpose and not disallow_transpose:
            # print(f"flipping the image")
            img, points = get_flipped_img_and_points(img, points)
            # print(f"transposing the image")
            img, points = get_transposed_img_and_points(img, points)
            already_transposed = True

        img_shape = img.shape[:2]

        if img_size is not None and size_differs(img_size, img_shape):
            img, points = resize_img_and_points(
                img, points, img_size, fractional=self.fractional
            )
            # except:
            # traceback.print_exc()
            # print(f'problem with resizing {self.image_path} realpath {self.realpath}, json_path {self.json_path} dtype & shape {img.dtype} {img.shape}')
            # sys.exit()

            img_shape = img.shape[:2]

        if augment:
            if do_flip is True and not already_transposed:
                # print(f"flipping the image")
                img, points = get_flipped_img_and_points(img, points)

            if (
                do_transpose is True
                and not already_transposed
                and not disallow_transpose
            ):
                # print(f"transposing the image {already_transposed} {require_transpose} {disallow_transpose}")
                img, points = get_transposed_img_and_points(img, points)
                img_shape = img.shape[:2]

            if do_transform is True:
                img, points = get_transformed_img_and_points(
                    img, points, verbose=verbose
                )

            if do_shift is True:
                img, points = get_shifted_img_and_points(img, points, verbose=verbose)
            #     img, points = get_transformed_img_and_points(img, points)

            if (
                do_swap_backgrounds
                and "foreground" in self.labels
                and new_background is not None
            ):
                foreground = self._get_maps(
                    points, img_shape, exclusive_label="foreground"
                )
                img = swap_backgrounds(
                    img,
                    foreground,
                    new_background,
                    img_shape,
                )

            # if do_random_brightness or do_random_contrast:
            #     # https://docs.opencv.org/4.x/d3/dc1/tutorial_basic_linear_transform.html
            #     alpha, beta = 1.0, 0.0
            #     if do_random_brightness:
            #         beta = 50 * (random.random())
            #     if do_random_contrast:
            #         alpha += random.random()
            #     img = cv.convertScaleAbs(img, alpha=alpha, beta=beta)

            if do_random_gamma:
                img = get_gamma_image(img, verbose=verbose)

            if do_random_blur:
                img = get_blurred_image(img, verbose=verbose)

            if do_random_noise:
                img = get_noisy_image(img, verbose=verbose)

            if do_random_channel_shift and not do_black_and_white:
                img = img[
                    :,
                    :,
                    random.choice(
                        [[1, 0, 2], [1, 2, 0], [0, 2, 1], [2, 0, 1], [2, 1, 0]]
                    ),
                ]

            if do_black_and_white:
                img = get_gray(img)

        return img, points


def plot_keypoints(s, radius=11):
    npp = s.get_aoi_keypoints()
    # npp["origin"] = s._get_origin()
    # lipp = s._check_lipp()
    # projection =
    pylab.figure()
    pylab.title("pca keypoints")
    pylab.imshow(s.get_image())
    for key in npp:
        draw_point(
            npp[key], color=sns.xkcd_rgb[named_points_colors[key]], radius=radius
        )

    npp2 = s.get_aoi_keypoints(label="pin")
    for key in npp2:
        draw_point(
            npp2[key], color=sns.xkcd_rgb[named_points_colors[key]], radius=radius
        )

    pylab.show()


def plot_targets(s, target="crystal"):
    fh = s.get_flat_hierarchy()

    image = s.get_image()
    masks = s.get_masks()
    kp1 = s.get_keypoints(global_keypoints[1])
    voronoi1 = s.get_voronoi(kp1)
    kpcentr1 = s.get_keypoint_centerness(list(kp1.values()))
    # kp2 = s.get_keypoints(global_keypoints[2])
    # voronoi2 = s.get_voronoi(kp2)
    # kpcentr2 = s.get_keypoint_centerness(list(kp2.values()))

    if target in masks:
        # centerness = s.get_centerness()[target]
        dt = s.get_distance_transform()[target]
        # idt = s.get_inverse_distance_transform()[target]
        # pdt = s.get_power_distance_transform()[target]
        # pidt = s.get_power_inverse_distance_transform()[target]
        # sdt = s.get_sqrt_distance_transform()[target]
        # sidt = s.get_sqrt_inverse_distance_transform()[target]
        target_mask = masks[target]
        ltrb_target = get_universal_ltrb(target_mask)
        l_target = ltrb_target[:, :, 0]
        t_target = ltrb_target[:, :, 1]
        r_target = ltrb_target[:, :, 2]
        b_target = ltrb_target[:, :, 3]
        unmask = get_unmasked_image(image.copy(), masks, target)
        bbox_mask = s.get_bbox_masks()[target]
        bbunmask = _get_unmasked_image(image.copy(), bbox_mask)
        ltrb_bbox = get_universal_ltrb(bbox_mask)
        l_bbox = ltrb_bbox[:, :, 0]
        t_bbox = ltrb_bbox[:, :, 1]
        r_bbox = ltrb_bbox[:, :, 2]
        b_bbox = ltrb_bbox[:, :, 3]
        aether_mask = np.logical_not(target_mask)
        aedt = (
            get_distance_transform(
                aether_mask.astype("uint8"), invert=True, normalize=False
            )
            / 2
            - 1
        )
        aedt[target_mask.astype(bool)] = dt[target_mask.astype(bool)]
        bbdt = get_distance_transform(bbox_mask.astype("uint8"))
        aether_bbox_mask = np.logical_not(bbox_mask)
        aebbdt = (
            get_distance_transform(
                aether_bbox_mask.astype("uint8"), invert=True, normalize=False
            )
            / 2
            - 1
        )
        aebbdt[bbox_mask.astype(bool)] = bbdt[bbox_mask.astype(bool)]
    else:
        target_mask = None
        centerness = None

    h = s.get_flat_hierarchy()

    fig, axes = pylab.subplots(5, 4)

    fig.set_tight_layout(True)
    a = axes.flatten()
    for aa in a:
        aa.set_axis_off()

    k = 0
    a[k].imshow(s.get_image())
    a[k].set_title("input image")

    k += 1
    a[k].imshow(s.get_flat_hierarchy())
    a[k].set_title("hierarchy")

    k += 1
    a[k].imshow(voronoi1)
    a[k].set_title("voronoi 1")

    k += 1
    a[k].imshow(kpcentr1)
    a[k].set_title("keypoints 1 centerness")

    # a[4].imshow(voronoi2)
    # a[4].set_title("voronoi 2")

    # a[5].imshow(kpcentr2)
    # a[5].set_title("keypoints 2 centerness")

    k_start = k + 1
    if target_mask is not None:
        for k, (i, d) in enumerate(
            zip(
                [
                    target_mask,
                    unmask,
                    # centerness,
                    dt,
                    aedt,
                    l_target,
                    t_target,
                    r_target,
                    b_target,
                    # pdt,
                    # sdt,
                    # idt,
                    # pidt,
                    # sidt,
                    bbox_mask,
                    bbunmask,
                    bbdt,
                    aebbdt,
                    l_bbox,
                    t_bbox,
                    r_bbox,
                    b_bbox,
                ],
                [
                    "mask",
                    "unmask",
                    # "centerness",
                    "dt",
                    "aether dt",
                    "target l",
                    "target t",
                    "target r",
                    "target b",
                    # "pdt",
                    # "sdt",
                    # "idt",
                    # "pidt",
                    # "sidt",
                    "bbox_mask",
                    "bbunmask",
                    "bbdt",
                    "aether bbdt",
                    "bbox l",
                    "bbox t",
                    "bbox r",
                    "bbox b",
                ],
            )
        ):
            a[k + k_start].imshow(i)
            a[k + k_start].set_title(f"{target} {d}")

    pylab.show()


def plot_voronoi(s):
    image = s.get_image()
    kps = [s.get_keypoints(global_keypoints[k]) for k in [1, 2]]
    kps.append(s.get_gang_of_five())
    for k, kp in enumerate(kps):
        pylab.figure()
        pylab.title(f"keypoints {k}")
        pylab.imshow(image)
        ax = pylab.gca()
        print(f"kp {kp}")
        for p, v in kp.items():
            try:
                color = sns.xkcd_rgb[named_points_colors[p]]
            except:
                color = "red"
            draw_point(v, ax=ax, color=color, radius=11)

        pylab.figure()
        pylab.title(f"voronoi {k}")
        print(f"keypoints for voronoi {kp}")
        pylab.imshow(s.get_voronoi(kp))

        pylab.figure()
        pylab.title(f"centerness {k}")
        pylab.imshow(s.get_keypoint_centerness(list(kp.values())))

    pylab.show()


def test():
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-j",
        "--json",
        default="examples/soleil_proxima_dataset/autocenter_100161_Wed_Jan_27_12:21:02_2021_bright_failed.json",
        # "soleil_proxima_dataset/100161_Wed_Feb__6_202122_2019_manual_omega_210.00_zoom_9_y_486_x_670.json"
        type=str,
        help="path to the json file containing sample annotation",
    )
    parser.add_argument(
        "-t",
        "--target",
        default="area_of_interest",
        type=str,
        help="target",
    )
    args = parser.parse_args()
    print("args", args)

    s = Sample(args.json)
    # plot_keypoints(s)
    # plot_voronoi(s)
    plot_targets(s, args.target)


if __name__ == "__main__":
    test()
