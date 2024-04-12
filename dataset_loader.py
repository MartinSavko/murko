#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import os
import re
import math
import glob
import random
import numpy as np
from skimage.transform import resize
from tensorflow import keras
from keras.utils import to_categorical
from keras.preprocessing import image
from keras.preprocessing.image import (
    save_img,
    load_img,
    img_to_array,
    array_to_img,
)

def get_hierarchical_mask_from_target(
    target,
    notions=[
        "crystal",
        "ice",
        "loop_inside",
        "loop",
        "pin",
        "stem",
        "capillary",
        "foreground",
    ],
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "capillary": 5,
        "ice": 6,
        "foreground": 7,
    },
    notions_order = ["crystal",
                     "ice",
                     "loop_inside",
                     "loop",
                     "pin",
                     "stem",
                     "capillary",
                     "foreground"],
):
    hierarchical_mask = np.zeros(target.shape[:2], dtype=np.uint8)
    k=0
    for notion in notions_order[::-1]:
        if notion in notions:
            k += 1
            l = notion_indices[notion]
            mask = target[:, :, l]
            if np.any(mask):
                hierarchical_mask[mask == 1] = k
    return hierarchical_mask


def get_img_size_as_scale_of_pixel_budget(
    scale, pixel_budget=768 * 992, ratio=0.75, modulo=32
):
    n = math.floor(math.sqrt(pixel_budget / ratio))
    new_n = n * scale
    img_size = np.array((new_n * ratio, new_n)).astype(int)
    img_size -= np.mod(img_size, modulo)
    return tuple(img_size)

def get_img_size(resize_factor, original_size=(1024, 1360), modulo=32):
    new_size = resize_factor * np.array(original_size)
    new_size = get_closest_working_img_size(new_size, modulo=modulo)
    return new_size

def get_closest_working_img_size(img_size, modulo=32):
    closest_working_img_size = img_size - np.mod(img_size, modulo)
    return tuple(closest_working_img_size.astype(int))

def size_differs(original_size, img_size):
    return original_size[0] != img_size[0] or original_size[1] != img_size[1]

def get_dynamic_batch_size(img_size, pixel_budget=768 * 992):
    return max(int(pixel_budget / np.prod(img_size)), 1)

def get_batch(i, img_paths, batch_size):
    half, r = divmod(batch_size, 2)
    indices = np.arange(i - half, i + half + r)
    return [img_paths[divmod(item, len(img_paths))[1]] for item in indices]

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def get_transposed_img_and_target(img, target):
    new_axes_order = (
        1,
        0,
    ) + tuple(range(2, len(img.shape)))
    img = np.transpose(img, new_axes_order)
    new_axes_order = (
        1,
        0,
    ) + tuple(range(2, len(target.shape)))
    target = np.transpose(target, new_axes_order)  # [:len(target.shape)])
    return img, target

def get_flipped_img_and_target(img, target):
    axis = random.choice([0, 1])
    img = flip_axis(img, axis)
    target = flip_axis(target, axis)
    return img, target

def get_transformed_img_and_target(
    img,
    target,
    default_transform_gang=[0, 0, 0, 0, 1, 1],
    zoom_factor=0.25,
    shift_factor=0.25,
    shear_factor=25,
    size=(512, 512),
    rotate_probability=0.5,
    shift_probability=0.5,
    shear_probability=0.5,
    zoom_probability=0.5,
    theta_min=-30.0,
    theta_max=30.0,
    resize=False,
):
    if resize:
        img = resize(img, size, anti_aliasing=True)
        target = resize(target, size, anti_aliasing=True)
    theta, tx, ty, shear, zx, zy = default_transform_gang
    size_y, size_x = img.shape[:2]
    # rotate
    if random.random() < rotate_probability:
        theta = random.uniform(theta_min, theta_max)
    # shear
    if random.random() < shear_probability:
        shear = random.uniform(-shear_factor, +shear_factor)
    # shift
    if random.random() < shift_probability:
        tx = random.uniform(-shift_factor * size_x, +shift_factor * size_x)
        ty = random.uniform(-shift_factor * size_y, +shift_factor * size_y)
    # zoom
    if random.random() < zoom_probability:
        zx = random.uniform(1 - zoom_factor, 1 + zoom_factor)
        zy = zx

    if np.any(np.array([theta, tx, ty, shear, zx, zy]) != default_transform_gang):
        transform_arguments = {
            "theta": theta,
            "tx": tx,
            "ty": ty,
            "shear": shear,
            "zx": zx,
            "zy": zy,
            "col_axis": 0,
            "row_axis": 1,
            "channel_axis": 2,
            "fill_mode": "constant",
            "cval": 0,
        }
        img = image.apply_affine_transform(img, **transform_arguments)
        target = image.apply_affine_transform(target, **transform_arguments)
        #target = image.apply_affine_transform(target.astype(np.float32), fill_mode='constant', cval=0, **transform_arguments)
        #target = np.astype(np.uint8)
    return img, target

class MultiTargetDataset(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        img_size,
        img_paths,
        background_paths="./soleil_proxima_dataset_v0.1/Backgrounds",
        img_string="img.jpg",
        label_string="masks.npy",
        click_string="user_click.npy",
        augment=False,
        transform=True,
        transpose=True,
        flip=True,
        swap_backgrounds=True,
        zoom_factor=0.25,
        shift_factor=0.25,
        shear_factor=45,
        default_transform_gang=[0, 0, 0, 0, 1, 1],
        scale_click=False,
        click_radius=320e-3,
        min_scale=0.15,
        max_scale=1.0,
        dynamic_batch_size=False,
        number_batch_size_scales=32,
        possible_ratios=[0.75, 1.0],
        pixel_budget=768 * 992,
        artificial_size_increase=1,
        notions=[
            "crystal",
            "loop_inside",
            "loop",
            "stem",
            "pin",
            "capillary",
            "ice",
            "foreground",
            "click",
        ],
        notion_indices={
            "crystal": 0,
            "loop_inside": 1,
            "loop": 2,
            "stem": 3,
            "pin": 4,
            "capillary": 5,
            "ice": 6,
            "foreground": 7,
        },
        shuffle_at_0=False,
        click="segmentation",
        target=True,
        black_and_white=True,
        random_brightness=True,
        random_channel_shift=False,
        verbose=False,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        if artificial_size_increase > 1:
            self.img_paths = img_paths * int(artificial_size_increase)
        else:
            self.img_paths = img_paths
        self.nimages = len(self.img_paths)
        self.img_string = img_string
        self.label_string = label_string
        self.click_string = click_string
        self.augment = augment
        self.transform = transform
        self.transpose = transpose
        self.flip = flip
        self.swap_backgrounds = swap_backgrounds
        self.zoom_factor = zoom_factor
        self.shift_factor = shift_factor
        self.shear_factor = shear_factor
        self.default_transform_gang = np.array(default_transform_gang)
        self.scale_click = scale_click
        self.click_radius = click_radius
        self.dynamic_batch_size = dynamic_batch_size
        if self.dynamic_batch_size:
            self.batch_size = 1
        self.possible_scales = np.linspace(
            min_scale, max_scale, number_batch_size_scales
        )
        self.possible_ratios = possible_ratios
        self.pixel_budget = pixel_budget
        self.notions = notions
        self.notion_indices = notion_indices
        self.candidate_backgrounds = {}
        self.batch_img_paths = []
        if self.swap_backgrounds:
            backgrounds = glob.glob(os.path.join(background_paths, "*.jpg"))
            for img_path in backgrounds:
                zoom = int(re.findall(".*_zoom_([\\d]*).*", img_path)[0])
                background = load_img(img_path)
                background = img_to_array(background, dtype="float32") / 255.0
                if zoom in self.candidate_backgrounds:
                    self.candidate_backgrounds[zoom].append(background)
                else:
                    self.candidate_backgrounds[zoom] = [background]
        self.shuffle_at_0 = shuffle_at_0
        self.click = click
        self.target = target
        self.black_and_white = black_and_white
        self.random_brightness = random_brightness
        self.random_channel_shift = random_channel_shift
        self.identity = False
        if "identity" in notions:
            self.identity = True
        if "hierarchy" in notions:
            self.hierarchy = True
            self.hierarchy_notions = [notion for notion in self.notions if notion not in ["hierarchy", "identity"]]
            self.hierarchy_num_classes = len(self.hierarchy_notions) + 1
        self.verbose = verbose

    def __len__(self):
        return math.ceil(len(self.img_paths) / self.batch_size)

    def consider_click(self):
        if self.target and "click" in self.notions:
            return True
        return False

    def __getitem__(self, idx):
        if idx == 0 and self.shuffle_at_0:
            random.Random().shuffle(self.img_paths)
        if self.dynamic_batch_size:
            img_size = get_img_size_as_scale_of_pixel_budget(
                random.choice(self.possible_scales),
                pixel_budget=self.pixel_budget,
                ratio=random.choice(self.possible_ratios),
            )
            batch_size = get_dynamic_batch_size(
                img_size, pixel_budget=self.pixel_budget
            )
            i = idx
            self.batch_img_paths = get_batch(i, self.img_paths, batch_size)
        else:
            img_size = self.img_size[:]
            batch_size = self.batch_size
            i = idx * self.batch_size
            start_index = i
            end_index = i + batch_size
            self.batch_img_paths = self.img_paths[start_index:end_index]

        final_img_size = img_size[:]
        batch_size = len(
            self.batch_img_paths
        )  # this handles case at the very last step ...

        do_flip = False
        do_transpose = False
        do_transform = False
        do_swap_backgrounds = False
        do_black_and_white = False
        do_random_brightness = False
        do_random_channel_shift = False
        if self.augment:
            if self.transform and random.random() < 0.5:
                do_transform = True
                if self.verbose:
                    print("do_transform")
            if self.transpose and random.random() < 0.5:
                final_img_size = img_size[::-1]
                do_transpose = True
                if self.verbose:
                    print("do_transpose")
                if self.flip and random.random() < 0.5:
                    do_flip = True
                    if self.verbose:
                        print("do_flip")
            else:
                if self.flip and random.random() < 0.5:
                    do_flip = True
                    if self.verbose:
                        print("do_flip")
            if self.swap_backgrounds and random.random() < 0.25:
                do_swap_backgrounds = True
                if self.verbose:
                    print("do_swap_backgrounds")
            if self.black_and_white and random.random() < 0.25:
                do_black_and_white = True
                if self.verbose:
                    print("do_black_and_white")
            if self.random_brightness and random.random() < 0.25:
                do_random_brightness = True
                if self.verbose:
                    print("do_random_brightness")
            if (
                not do_black_and_white
                and self.random_channel_shift
                and random.random() < 0.25
            ):
                do_random_channel_shift = True
                if self.verbose:
                    print("do_random_channel_shift")

        x = np.zeros((batch_size,) + final_img_size + (3,), dtype="float32")
        y = []
        for notion in self.notions:
            if 'click' not in notion and notion not in ["identity", "hierarchy"]:
                y.append(np.zeros((batch_size,) + final_img_size + (1,), dtype="uint8"))
            elif notion == "identity":
                y.append(np.zeros((batch_size,) + final_img_size + (1,), dtype="float32"))
            elif notion == "hierarchy":
                y.append(np.zeros((batch_size,) + final_img_size + (self.hierarchy_num_classes,), dtype="float32"))
                #y.append(np.zeros((batch_size,) + final_img_size + (1,), dtype="uint8"))
            elif click == 'click_segmentation':
                y.append(
                    np.zeros((batch_size,) + final_img_size + (1,), dtype="float32")
                )
            else:
                 y.append(np.zeros((batch_size,) + (3,), dtype="float32"))
       
        for j, img_path in enumerate(self.batch_img_paths):
            resize_factor = 1.0
            original_image = load_img(img_path)
            original_size = original_image.size[::-1]
            img = img_to_array(original_image, dtype="float32") / 255.0
            masks_name = img_path.replace(self.img_string, self.label_string)
            user_click_name = img_path.replace(self.img_string, self.click_string)
            if self.target:
                target = np.load(masks_name)
            try:
                zoom = int(re.findall(".*_zoom_([\\d]*).*", img_path)[0])
            except BaseException:
                zoom = 1

            if size_differs(original_size, img_size):
                resize_factor = np.array(img_size) / np.array(original_size)

            if self.consider_click():
                user_click = np.array(np.load(user_click_name)).astype("float32")
                click_present = all(user_click[:2] >= 0)
                if self.augment and click_present:
                    click_mask = get_cpi_from_user_click(
                        user_click,
                        target.shape[:2],
                        1.0,
                        img_path,
                        click_radius=self.click_radius,
                        zoom=zoom,
                        scale_click=self.scale_click,
                    )
                    target = np.concatenate([target, click_mask], axis=2)
                elif click_present:
                    user_click_frac = user_click * resize_factor
                else:
                    user_click_frac = np.array([np.nan, np.nan])

            if self.target and np.all(
                target[:, :, self.notions.index("foreground")] == 0
            ):
                do_swap_backgrounds = False

            if do_transpose is True:
                img, target = get_transposed_img_and_target(img, target)

            if do_flip is True:
                img, target = get_flipped_img_and_target(img, target)

            if do_transform is True:
                img, target = get_transformed_img_and_target(
                    img,
                    target,
                    zoom_factor=self.zoom_factor,
                    shift_factor=self.shift_factor,
                    shear_factor=self.shear_factor,
                )

            if do_swap_backgrounds is True and "background" not in img_path:
                new_background = random.choice(self.candidate_backgrounds[zoom])
                if size_differs(img.shape[:2], new_background.shape[:2]):
                    new_background = resize(
                        new_background, img.shape[:2], anti_aliasing=True
                    )
                img[
                    target[:, :, self.notions.index("foreground")] == 0
                ] = new_background[target[:, :, self.notions.index("foreground")] == 0]

            if do_random_brightness is True:
                img = image.random_brightness(img, [0.75, 1.25]) / 255.0

            if do_random_channel_shift is True:
                img = image.random_channel_shift(img, 0.5, channel_axis=2)

            if size_differs(img.shape[:2], final_img_size):
                img = resize(img, final_img_size, anti_aliasing=True)
                if self.target:
                    target = resize(
                        target.astype("float32"),
                        final_img_size,
                        mode="constant",
                        cval=0,
                        anti_aliasing=False,
                        preserve_range=True,
                    )
                    
            if do_black_and_white or self.identity:
                img_bw = img.mean(axis=2)

            if self.augment and self.consider_click() and click_present:
                transformed_click = target[:, :, -1]
                user_click = np.unravel_index(
                    np.argmax(transformed_click), transformed_click.shape
                )[:2]
                user_click_frac = np.array(user_click) / np.array(final_img_size)
                if self.click == "click_segmentation":
                    cpi = get_cpi_from_user_click(
                        user_click,
                        final_img_size,
                        resize_factor,
                        img_path + "augment",
                        click_radius=self.click_radius,
                        zoom=zoom,
                        scale_click=self.scale_click,
                    )
                    target[:, :, -1] = cpi[:, :, 0]

            if self.consider_click() and self.click == "click_segmentation":
                target[:, :, :-1] = (target[:, :, :-1] > 0.5).astype("uint8")
            elif self.consider_click() and self.click == "click_regression":
                click_present = int(click_present)
                y_click, x_click = user_click_frac
            elif self.target:
                target = (target > 0.5).astype("uint8")
            
            if do_black_and_white:
                img = np.stack([img_bw] * 3, axis=2)
            
            if self.hierarchy:
                #print('target', target.shape, target.min(), target.max())
                hierarchy = get_hierarchical_mask_from_target(target, notions=self.hierarchy_notions, notion_indices=self.notion_indices)
                #print('hierarchy', hierarchy.shape, hierarchy.min(), hierarchy.max())
                hierarchy = to_categorical(hierarchy, num_classes=self.hierarchy_num_classes)
                #print('categorical', hierarchy.shape, hierarchy.min(), hierarchy.max())
            x[j] = img
            if self.target:
                for k, notion in enumerate(self.notions):
                    if notion in self.notion_indices:
                        l = self.notion_indices[notion]
                        if l != -1:
                            y[k][j] = np.expand_dims(target[:, :, l], axis=2)
                        elif l == -1 and self.click == "click_segmentation":
                            y[k][j] = np.expand_dims(target[:, :, l], axis=2)
                        elif l == -1 and self.click == "click_regression":
                            y[k][j] = np.array([click_present, y_click, x_click])
                    elif notion == "identity":
                        y[k][j] = np.expand_dims(img_bw, axis=2)
                    elif notion == "hierarchy":
                        y[k][j] = hierarchy
        if self.target and len(y) == 1:
            y = y[0]
        if self.target:
            return x, y
        else:
            return x


class SampleSegmentationDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(
        self,
        batch_size,
        img_size,
        input_img_paths,
        img_string="img.jpg",
        label_string="foreground.png",
        augment=False,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.img_string = img_string
        self.label_string = label_string
        self.augment = augment
        self.zoom_factor = 0.2
        self.shift_factor = 0.25
        self.shear_factor = 15
        self.default_transform_gang = np.array([0, 0, 0, 0, 1, 1])

    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        self.batch_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_paths = [
            path.replace(self.img_string, self.label_string)
            for path in self.batch_img_paths
        ]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, (img_path, target_path) in enumerate(
            zip(batch_input_img_paths, batch_target_img_paths)
        ):
            img = (
                img_to_array(
                    load_img(img_path, target_size=self.img_size), dtype="float32"
                )
                / 255.0
            )
            target = load_ground_truth_image(target_path, target_size=self.img_size)

            if self.augment:
                img, target = get_transformed_img_and_target(
                    img,
                    target,
                    zoom_factor=self.zoom_factor,
                    shift_factor=self.shift_factor,
                    shear_factor=self.shear_factor,
                )
            x[j] = img
            y[j] = target
        return x, y


class CrystalClickDataset(keras.utils.Sequence):
    def __init__(
        self,
        batch_size,
        img_size,
        img_paths,
        augment=False,
        transpose=True,
        flip=True,
        zoom_factor=0.2,
        shift_factor=0.25,
        shear_factor=15,
        default_transform_gang=[0, 0, 0, 0, 1, 1],
        scale_click=False,
        click_radius=320e-3,
        min_scale=0.15,
        max_scale=1.0,
        dynamic_batch_size=True,
        number_batch_size_scales=32,
        possible_ratios=[0.75, 1.0],
        pixel_budget=768 * 992,
    ):
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.augment = augment
        self.transpose = transpose
        self.flip = flip
        self.zoom_factor = zoom_factor
        self.shift_factor = shift_factor
        self.shear_factor = shear_factor
        self.default_transform_gang = np.array(default_transform_gang)
        self.scale_click = scale_click
        self.click_radius = click_radius
        self.dynamic_batch_size = dynamic_batch_size
        self.possible_scales = np.linspace(
            min_scale, max_scale, number_batch_size_scales
        )
        self.possible_ratios = possible_ratios
        self.pixel_budget = pixel_budget

        if self.dynamic_batch_size:
            self.batch_size = 1

    def __len__(self):
        return math.ceil(len(self.img_paths) / self.batch_size)

    def __getitem__(self, idx):
        if idx == 0:
            random.Random().shuffle(self.img_paths)
        if self.dynamic_batch_size:
            img_size = get_img_size_as_scale_of_pixel_budget(
                random.choice(self.possible_scales),
                pixel_budget=self.pixel_budget,
                ratio=random.choice(self.possible_ratios),
            )
            batch_size = get_dynamic_batch_size(
                img_size, pixel_budget=self.pixel_budget
            )
            i = idx
        else:
            img_size = self.img_size
            batch_size = self.batch_size
            i = idx * self.batch_size

        final_img_size = img_size[:]
        batch_img_paths = self.img_paths[i : i + batch_size]
        batch_size = len(batch_img_paths)  # this handles case at the very last step ...

        if self.augment:
            do_transpose = False
            if self.transpose and random.random() > 0.5:
                final_img_size = img_size[::-1]
                do_transpose = True
            do_flip = False
            if self.flip and random.random() > 0.5:
                do_flip = True

        x = np.zeros((batch_size,) + final_img_size + (3,), dtype="float32")
        y = np.zeros((batch_size,) + final_img_size + (1,), dtype="float32")
        for j, img_path in enumerate(batch_img_paths):
            user_click = None
            try:
                original_image = load_img(img_path)
                original_size = original_image.size[::-1]
                img = img_to_array(original_image, dtype="float32") / 255.0
                if np.any(np.isnan(img)):
                    os.system(
                        "echo this gave nan, please check %s >> click_generation_problems_new.txt"
                        % img_path
                    )
                    continue
                if original_size[0] > original_size[1]:
                    original_size = original_size[::-1]
                    img = np.reshape(img, original_size + img.shape[2:])
                    os.system(
                        "echo wrong ratio, please check %s >> click_generation_problems_new.txt"
                        % img_path
                    )
                img = resize(img, final_img_size)
            except BaseException:
                print(traceback.print_exc())
                os.system(
                    "echo load_img failed %s >> click_generation_problems_new.txt"
                    % img_path
                )
                img = np.zeros(img_size + (3,))
                original_size = img_size[:]
                user_click = np.array([-1, -1])

            try:
                zoom = int(re.findall(".*_zoom_([\\d]*).*", img_path)[0])
            except BaseException:
                zoom = 1
            if os.path.basename(img_path) == "img.jpg" and user_click is None:
                user_click = np.load(img_path.replace("img.jpg", "user_click.npy"))
            elif "shapes_of_background" in img_path:
                user_click = np.array([-1.0, -1.0])
            else:
                try:
                    user_click = np.array(
                        list(
                            map(
                                float,
                                re.findall(".*_y_([-\\d]*)_x_([-\\d]*).*", img_path)[0],
                            )
                        )
                    )
                except BaseException:
                    user_click = np.array([-1.0, -1.0])

            resize_factor = np.array([1.0, 1.0])
            if original_size[0] != img_size[0] and original_size[1] != img_size[1]:
                resize_factor = np.array(img_size) / np.array(original_size)

            user_click *= resize_factor
            user_click = user_click.astype("float32")
            cpi = get_cpi_from_user_click(
                user_click,
                final_img_size,
                resize_factor,
                img_path,
                click_radius=self.click_radius,
                zoom=zoom,
                scale_click=self.scale_click,
            )
            if cpi is None:
                continue
            if self.augment:
                if do_transpose is True:
                    img, cpi = get_transposed_img_and_target(img, cpi)
                if do_flip is True:
                    img, cpi = get_flipped_img_and_target(img, cpi)
                img, cpi = get_transformed_img_and_target(
                    img,
                    cpi,
                    zoom_factor=self.zoom_factor,
                    shift_factor=self.shift_factor,
                    shear_factor=self.shear_factor,
                )

                if all(user_click[:2] >= 0):
                    user_click = np.unravel_index(np.argmax(cpi), cpi.shape)
                cpi = get_cpi_from_user_click(
                    user_click[:2],
                    final_img_size,
                    resize_factor,
                    img_path + "augment",
                    click_radius=self.click_radius,
                    zoom=zoom,
                    scale_click=self.scale_click,
                )

            x[j] = img
            y[j] = cpi

        return x, y

