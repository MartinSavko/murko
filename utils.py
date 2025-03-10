#!/usr/bin/env python
# -*- coding: utf-8 -*-

from skimage.morphology import remove_small_objects
from skimage.measure import regionprops
from skimage.transform import resize

import os
import time
import traceback
import pylab
import keras
from keras.preprocessing.image import (
    save_img,
    load_img,
    img_to_array,
    array_to_img,
)
import scipy.ndimage as ndi
import numpy as np
import random
from dataset_loader import (
    get_hierarchical_mask_from_target,
    get_transposed_img_and_target,
    get_transformed_img_and_target,
    get_flipped_img_and_target,
)

# calibrations = {
# 1: np.array([0.00160829, 0.001612]),
# 2: np.array([0.00129349, 0.0012945]),
# 3: np.array([0.00098891, 0.00098577]),
# 4: np.array([0.00075432, 0.00075136]),
# 5: np.array([0.00057437, 0.00057291]),
# 6: np.array([0.00043897, 0.00043801]),
# 7: np.array([0.00033421, 0.00033406]),
# 8: np.array([0.00025234, 0.00025507]),
# 9: np.array([0.00019332, 0.00019494]),
# 10: np.array([0.00015812, 0.00015698]),
# }


def generate_click_loss_and_metric_figures(
    click_radius=360e-3, image_shape=(1024, 1360), nclicks=10, ntries=1000, display=True
):
    resize_factor = np.array(image_shape) / np.array((1024, 1360))
    distances = []
    bfcs = []
    bio1 = []
    bio1m = keras.metrics.BinaryIoUm(target_class_ids=[1], threshold=0.5)
    bio0 = []
    bio0m = keras.metrics.BinaryIoUm(target_class_ids=[0], threshold=0.5)
    biob = []
    biobm = keras.metrics.BinaryIoUm(target_class_ids=[0, 1], threshold=0.5)
    concepts = {
        "bfcs": bfcs,
        "bio1": bio1,
        "bio0": bio0,
        "biob": biob,
        "distances": distances,
    }

    for k in range(nclicks):
        click = (
            np.array(image_shape)
            * np.random.rand(
                2,
            )
        ).astype(int)
        cpi_true = click_probability_image(
            click[1],
            click[0],
            image_shape,
            click_radius=click_radius,
            resize_factor=resize_factor,
            scale_click=False,
        )
        cpi_true = np.expand_dims(cpi_true, (0, -1))
        for n in range(ntries // nclicks):
            tclick = (
                np.array(image_shape)
                * np.random.rand(
                    2,
                )
            ).astype(int)
            cpi_pred = click_probability_image(
                tclick[1],
                tclick[0],
                image_shape,
                click_radius=click_radius,
                resize_factor=resize_factor,
                scale_click=False,
            )
            cpi_pred = np.expand_dims(cpi_pred, (0, -1))
            concepts["distances"].append(np.linalg.norm(click - tclick, 2))
            concepts["bfcs"].append(
                keras.losses.binary_focal_crossentropy(cpi_true, cpi_pred)
                .numpy()
                .mean()
            )
            bio1m.reset_state()
            bio1m.update_state(cpi_true, cpi_pred)
            concepts["bio1"].append(bio1m.result().numpy())
            bio0m.reset_state()
            bio0m.update_state(cpi_true, cpi_pred)
            concepts["bio0"].append(bio0m.result().numpy())
            biobm.reset_state()
            biobm.update_state(cpi_true, cpi_pred)
            concepts["biob"].append(biobm.result().numpy())

    for concept in concepts:
        concepts[concept] = np.array(concepts[concept])
    concepts["distances"] /= np.linalg.norm(image_shape, 2)
    concepts["bfcs"] /= concepts["bfcs"].max()
    pylab.figure(figsize=(16, 9))
    pylab.title(
        "image shape %dx%d, click_radius=%.3f"
        % (image_shape[0], image_shape[1], click_radius)
    )
    for concept in ["bfcs", "bio1", "bio0", "biob"]:
        pylab.plot(concepts["distances"], concepts[concept], "o", label=concept)
    pylab.xlabel("distances")
    pylab.ylabel("loss/metrics")
    pylab.savefig(
        "click_metric_cr_%.3f_img_shape_%dx%d.png"
        % (click_radius, image_shape[0], image_shape[1])
    )
    pylab.legend()
    if display:
        pylab.show()
    return concepts


class ClickMetric(keras.metrics.MeanAbsoluteError):
    def __init__(self, name="click_metric", dtype=None):
        super(ClickMetric, self).__init__(name=name, dtype=dtype)

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred, sample_weight)


class ClickLoss(keras.losses.MeanSquaredError):
    def call(self, ci_true, ci_pred):
        com_true = keras_center_of_mass(ci_true)
        com_pred = keras_center_of_mass(ci_pred)

        mse = super().call(com_true, com_pred)
        mse = replacenan(mse)
        bcl = keras.ops.mean(
            keras.losses.binary_crossentropy(ci_true, ci_pred), axis=(1, 2)
        )
        click_present = keras.ops.reshape(K.max(ci_true, axis=(1, 2)), (-1))
        total = bcl * (1 - click_present) + mse * (click_present)

        return total


def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return np.exp(
        -((x - mx) ** 2.0 / (2.0 * sx**2.0) + (y - my) ** 2.0 / (2.0 * sy**2.0))
    )


def click_probability_image(
    click_x,
    click_y,
    img_shape,
    zoom=1,
    click_radius=320e-3,
    resize_factor=1.0,
    scale_click=True,
):
    x = np.arange(0, img_shape[1], 1)
    y = np.arange(0, img_shape[0], 1)
    x, y = np.meshgrid(x, y)
    if scale_click:
        mmppx = calibrations[zoom] / resize_factor
    else:
        mmppx = calibrations[1] / resize_factor
    sx = click_radius / mmppx.mean()
    sy = sx
    z = gauss2d(x, y, mx=click_x, my=click_y, sx=sx, sy=sy)
    return z


def replacenan(t):
    return keras.ops.where(keras.ops.isnan(t), keras.ops.zeros_like(t), t)


def click_loss(ci_true, ci_pred):
    total = keras.losses.mean_squared_error(ci_true, ci_pred)
    return total


def keras_center_of_mass(image_batch, threshold=0.5):
    """https://stackoverflow.com/questions/51724450/finding-centre-of-mass-of-tensor-tensorflow"""
    print(image_batch.shape)
    keras.ops.cast(image_batch >= threshold, "float32")
    batch_size, height, width, depth = image_batch.shape
    # Make array of coordinates (each row contains three coordinates)

    ii, jj, kk = keras.ops.meshgrid(
        keras.ops.range(height),
        keras.ops.range(width),
        keras.ops.range(depth),
        indexing="ij",
    )
    coords = keras.ops.stack(
        [
            keras.ops.reshape(ii, (-1,)),
            keras.ops.reshape(jj, (-1,)),
            keras.ops.reshape(kk, (-1)),
        ],
        axis=-1,
    )
    coords = keras.ops.cast(coords, "float32")
    # Rearrange input into one vector per volume
    volumes_flat = keras.ops.reshape(image_batch, [-1, height * width, 1])
    # Compute total mass for each volume
    total_mass = keras.ops.sum(volumes_flat, axis=1)
    # Compute centre of mass
    centre_of_mass = keras.ops.sum(volumes_flat * coords, axis=1) / total_mass

    return centre_of_mass


def click_image_loss(ci_true, ci_pred):
    if K.max(ci_true) == 0:
        return keras.losses.binary_crossentropy(y_true, y_pred)
    y_true = centre_of_mass(ci_true)
    y_pred = centre_of_mass(ci_pred)
    return keras.losses.mean_squared_error(y_true, y_pred)


def click_mean_absolute_error(ci_true, ci_pred):
    if K.max(ci_pred) < 0.5 and K.max(ci_true) == 0:
        return 0

    y_true = centre_of_mass(ci_true)
    y_pred = centre_of_mass(ci_pred)
    return keras.losses.mean_absolute_error(y_true, y_pred)


def click_batch_loss(click_true_batch, click_pred_image_batch):
    return [
        click_loss(click_true, click_pred_image)
        for click_true, click_pred_image in zip(
            click_true_batch, click_pred_image_batch
        )
    ]


def get_click_from_single_click_image(click_image):
    click_pred = np.zeros((3,))
    m = click_image.max()
    click_pred[:2] = np.array(
        np.unravel_index(np.argmax(click_image), click_image.shape)[:2], dtype="float32"
    )
    click_pred[2] = m
    return click_pred


def get_clicks_from_click_image_batch(click_image_batch):
    input_shape = click_image_batch.shape
    output_shape = (input_shape[0], 3)
    click_pred = np.zeros(output_shape)
    for k, click_image in enumerate(click_image_batch):
        click_pred[k] = get_click_from_single_click_image(click_image)
    return click_pred


def display_target(target_array):
    normalized_array = (target_array.astype("uint8")) * 127
    pylab.axis("off")
    pylab.imshow(normalized_array[:, :, 0])


def get_dataset(batch_size, img_size, img_paths, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)


def augment_sample(
    img_path,
    img,
    target,
    user_click,
    do_swap_backgrounds,
    do_flip,
    do_transpose,
    zoom,
    candidate_backgrounds,
    notions,
    zoom_factor,
    shift_factor,
    shear_factor,
):
    if do_swap_backgrounds is True and "background" not in img_path:
        new_background = random.choice(candidate_backgrounds[zoom])
        if size_differs(img.shape[:2], new_background.shape[:2]):
            new_background = resize(new_background, img.shape[:2], anti_aliasing=True)
        img[target[:, :, notions.index("foreground")] == 0] = new_background[
            target[:, :, notions.index("foreground")] == 0
        ]

    if self.augment and do_transpose is True:
        img, target = get_transposed_img_and_target(img, target)

    if self.augment and do_flip is True:
        img, target = get_flipped_img_and_target(img, target)

    if self.augment:
        img, target = get_transformed_img_and_target(
            img,
            target,
            zoom_factor=zoom_factor,
            shift_factor=shift_factor,
            shear_factor=shear_factor,
        )

    return img, target


def get_img_and_target(img_path, img_string="img.jpg", label_string="masks.npy"):
    original_image = load_img(img_path)
    original_size = original_image.size[::-1]
    img = img_to_array(original_image, dtype="float32") / 255.0
    masks_name = img_path.replace(img_string, label_string)
    target = np.load(masks_name)
    return img, target


def get_img(img_path, size=(224, 224)):
    original_image = load_img(img_path)
    img = img_to_array(original_image, dtype="float32") / 255.0
    img = resize(img, size, anti_aliasing=True)
    return img


def plot_augment(
    img_path,
    ntransformations=14,
    figsize=(24, 16),
    zoom_factor=0.5,
    shift_factor=0.5,
    shear_factor=45,
    rotate_probability=1,
    shear_probability=1,
    zoom_probability=1,
    shift_probability=1,
):
    fig, axes = pylab.subplots((2 * ntransformations) // 6 + 1, 6)
    fig.set_size_inches(*figsize)
    title = get_title_from_img_path(img_path)
    fig.suptitle(title)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")
    img, target = get_img_and_target(img_path)
    ax[0].imshow(img)
    ax[0].set_title("input image")
    ax[1].imshow(get_hierarchical_mask_from_target(target))
    ax[1].set_title("original_target")

    for t in range(1, ntransformations + 1):
        wimg = img[::]
        wtarget = target[::]
        if random.random() > 0.5:
            wimg, wtarget = get_flipped_img_and_target(wimg, wtarget)
        if random.random() > 0.5:
            wimg, wtarget = get_transposed_img_and_target(wimg, wtarget)

        wimg, wtarget = get_transformed_img_and_target(
            wimg,
            wtarget,
            shear_factor=shear_factor,
            zoom_factor=zoom_factor,
            shift_factor=shift_factor,
            rotate_probability=rotate_probability,
            shear_probability=shear_probability,
            zoom_probability=zoom_probability,
            shift_probability=shift_probability,
        )

        ax[2 * t].imshow(wimg)
        ax[2 * t].set_title("%d input" % (t + 1))
        ax[2 * t + 1].imshow(get_hierarchical_mask_from_target(wtarget))
        ax[2 * t + 1].set_title("%d target" % (t + 1))

    pylab.show()


def get_title_from_img_path(img_path):
    return os.path.basename(os.path.dirname(img_path))


def plot(
    sample,
    title="",
    k=0,
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
):
    fig, axes = pylab.subplots(2, 5)
    fig.set_size_inches(24, 16)
    fig.suptitle(title)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")
    ax[0].imshow(sample[0][k])
    ax[0].set_title("input image")
    for l in range(len(sample[1])):
        ax[1 + l].imshow(sample[1][l][k][:, :, 0])
        ax[1 + l].set_title(notions[l])
    pylab.show()


def plot_batch(
    batch_size=16,
    transform=True,
    augment=True,
    swap_backgrounds=True,
    black_and_white=True,
    shuffle_at_0=True,
    flip=True,
    transpose=True,
    model_img_size=(256, 320),
    figsize=(24, 16),
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "capillary",
        "ice",
        "foreground",
        "hierarchy",
        "identity",
    ],
):
    paths, _ = get_training_and_validation_datasets(
        directory="images_and_labels", split=0.2
    )
    gen = get_generator(
        batch_size,
        model_img_size,
        paths,
        notions=notions,
        augment=augment,
        transform=transform,
        swap_backgrounds=swap_backgrounds,
        flip=flip,
        transpose=transpose,
        dynamic_batch_size=False,
        artificial_size_increase=False,
        shuffle_at_0=shuffle_at_0,
        black_and_white=black_and_white,
        verbose=True,
    )

    imgs, targets = gen[0]
    targets_as_multichannel_masks = np.zeros(
        imgs.shape[:3] + (len(gen.hierarchy_notions),), dtype="uint8"
    )
    for k in range(batch_size):
        for l in range(len(gen.hierarchy_notions)):
            targets_as_multichannel_masks[k, :, :, l] = targets[l][k, :, :, 0]
    fig, axes = pylab.subplots((2 * batch_size) // 6 + 1, 6)
    fig.set_size_inches(*figsize)
    title = "batch plot"
    fig.suptitle(title)
    ax = axes.flatten()
    for a in ax:
        a.axis("off")
    for t in range(batch_size):
        ax[2 * t].imshow(imgs[t])
        ax[2 * t].set_title("%d input" % (t))
        ax[2 * t + 1].imshow(
            get_hierarchical_mask_from_target(targets_as_multichannel_masks[t])
        )
        ax[2 * t + 1].set_title("%d target" % (t))
    pylab.show()


def get_generator(
    paths,
    batch_size=16,
    model_img_size=(320, 256),
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "capillary",
        "ice",
        "foreground",
        "hierarchy",
        "identity",
    ],
    augment=True,
    transform=True,
    swap_backgrounds=True,
    black_and_white=True,
    shuffle_at_0=True,
    flip=True,
    transpose=True,
    dynamic_batch_size=False,
    artificial_size_increase=False,
    verbose=True,
):
    gen = MultiTargetDataset(
        batch_size,
        model_img_size,
        paths,
        notions=notions,
        augment=augment,
        transform=transform,
        swap_backgrounds=swap_backgrounds,
        flip=flip,
        transpose=transpose,
        dynamic_batch_size=dynamic_batch_size,
        artificial_size_increase=artificial_size_increase,
        shuffle_at_0=shuffle_at_0,
        black_and_white=black_and_white,
        verbose=verbose,
    )
    return gen


def save_predictions(
    input_images,
    predictions,
    image_paths,
    ground_truths,
    notions,
    notion_indices,
    model_img_size,
    model_name="default",
    train=False,
    target=False,
    threshold=0.5,
    click_threshold=0.95,
):
    _start = time.time()
    for k, input_image in enumerate(input_images):
        hierarchical_mask = np.zeros(model_img_size, dtype=np.uint8)
        predicted_masks = np.zeros(model_img_size + (len(notions),), dtype=np.uint8)
        if "click" in notions:
            notions_in_order = notions[:-1][::-1] + [notions[-1]]
        else:
            notions_in_order = notions[::-1]
        for notion in notions_in_order:
            notion_value = notions.index(notion) + 1
            l = notion_indices[notion]
            if l != -1:
                mask = (predictions[l][k] > threshold)[:, :, 0]
                predicted_masks[:, :, l] = mask
            else:
                mask = (predictions[l][k] > click_threshold)[:, :, 0]
                predicted_masks[:, :, l] = mask
            if np.any(mask):
                hierarchical_mask[mask == 1] = notion_value
            hierarchical_mask[-1, -(1 + notions.index(notion))] = notion_value
        if target:
            label_mask = np.zeros(model_img_size, dtype=np.uint8)
            for notion in notions_in_order:
                notion_value = notions.index(notion) + 1
                l = notion_indices[notion]
                if l != -1:
                    mask = (ground_truths[l][k] > threshold)[:, :, 0]
                else:
                    mask = (predictions[l][k] > click_threshold)[:, :, 0]
                if np.any(mask):
                    label_mask[mask == 1] = notion_value
                label_mask[-1, -(1 + notions.index(notion))] = notion_value

        name = os.path.basename(image_paths[k])
        prefix = name[:-4]
        directory = os.path.dirname(image_paths[k])

        if train:
            prefix += "_train"

        template = "%s_%s_model_img_size_%dx%d" % (
            prefix,
            model_name.replace(".h5", ""),
            model_img_size[0],
            model_img_size[1],
        )

        prediction_img_path = os.path.join(
            directory, "%s_hierarchical_mask_high_contrast_predicted.png" % (template)
        )
        save_img(
            prediction_img_path, np.expand_dims(hierarchical_mask, axis=2), scale=True
        )

        predicted_masks_name = os.path.join(directory, "%s.npy" % template)
        np.save(predicted_masks_name, predicted_masks)

        if target:
            fig, axes = pylab.subplots(1, 3)
        else:
            fig, axes = pylab.subplots(1, 2)
        fig.set_size_inches(16, 9)
        title = name
        fig.suptitle(title)
        axes[0].set_title(
            "input image with predicted click and loop bounding box (if any)"
        )
        axes[0].imshow(input_image)
        axes[1].set_title("raw segmentation result with most likely click (if any)")
        axes[1].imshow(hierarchical_mask)
        if target:
            axes[2].set_title("ground truth")
            axes[2].imshow(label_mask)

        for a in axes.flatten():
            a.axis("off")

        original_shape = np.array(input_image.shape[:2])
        prediction_shape = np.array(hierarchical_mask.shape)
        most_likely_click = np.array(get_most_likely_click(predictions, k=k))
        if -1 not in most_likely_click:
            mlc_ii = most_likely_click * original_shape
            click_patch_ii = pylab.Circle(mlc_ii[::-1], radius=2, color="green")
            axes[0].add_patch(click_patch_ii)

            mlc_hm = most_likely_click * prediction_shape
            click_patch_hm = pylab.Circle(mlc_hm[::-1], radius=2, color="green")
            axes[1].add_patch(click_patch_hm)

        loop_present, r, c, h, w = get_loop_bbox(predictions, k=k)
        if loop_present != -1:
            r *= original_shape[0]
            c *= original_shape[1]
            h *= original_shape[0]
            w *= original_shape[1]
            C, R = int(c - w / 2), int(r - h / 2)
            W, H = int(w), int(h)
            loop_bbox_patch = pylab.Rectangle(
                (C, R), W, H, linewidth=1, edgecolor="green", facecolor="none"
            )
            axes[0].add_patch(loop_bbox_patch)

        comparison_path = prediction_img_path.replace(
            "hierarchical_mask_high_contrast_predicted", "comparison"
        )
        pylab.savefig(comparison_path)
        pylab.close()
        print("saving %s" % comparison_path)
    end = time.time()

    print(
        "%d predictions saved in %.4f seconds (%.4f per image)"
        % (len(input_images), end - _start, (end - _start) / len(input_images))
    )


def plot_analysis(
    input_images, analysis, figsize=(16, 9), model_name="default", image_paths=None
):
    _start = time.time()
    descriptions = analysis["descriptions"]

    for k, input_image in enumerate(input_images):
        hierarchical_mask = descriptions[k]["hierarchical_mask"]

        if image_paths is None or image_paths != []:
            name = os.path.basename(image_paths[k])
            prefix = name[:-4]
            directory = os.path.dirname(image_paths[k])
        else:
            name = "%.1f" % time.time()
            prefix = "test"
            directory = "/tmp"

        print("name, prefix, directory", name, prefix, directory)
        template = "%s_%s_model_img_size_%dx%d" % (
            prefix,
            name,
            hierarchical_mask.shape[0],
            hierarchical_mask.shape[1],
        )
        print("template", template)
        prediction_img_path = os.path.join(
            directory, "%s_hierarchical_mask_high_contrast_predicted.png" % (template)
        )
        # save_img(prediction_img_path, np.expand_dims(hierarchical_mask, axis=2), scale=True)

        # predicted_masks_name = os.path.join(directory, '%s.npy' % template)
        # np.save(predicted_masks_name, predicted_masks)

        fig, axes = pylab.subplots(1, 2)
        fig.set_size_inches(figsize)
        fig.suptitle(name)
        axes[0].set_title(
            "input image with predicted click and loop bounding box (if any)"
        )
        axes[0].imshow(input_image)
        axes[1].set_title("raw segmentation result with most likely click (if any)")
        axes[1].imshow(hierarchical_mask)

        for a in axes.flatten():
            a.axis("off")

        prediction_shape = np.array(hierarchical_mask.shape)
        most_likely_click = descriptions[k]["most_likely_click"]
        original_shape = descriptions[k]["original_shape"]

        if -1 not in most_likely_click:
            mlc_ii = most_likely_click * original_shape
            click_patch_ii = pylab.Circle(mlc_ii[::-1], radius=2, color="green")
            axes[0].add_patch(click_patch_ii)

            mlc_hm = most_likely_click * prediction_shape
            click_patch_hm = pylab.Circle(mlc_hm[::-1], radius=2, color="green")
            axes[1].add_patch(click_patch_hm)

        loop_present, r, c, h, w = descriptions[k]["aoi_bbox"]
        if loop_present != -1:
            try:
                r *= original_shape[0]
                c *= original_shape[1]
                h *= original_shape[0]
                w *= original_shape[1]
                C, R = int(c - w / 2), int(r - h / 2)
                W, H = int(w), int(h)
                loop_bbox_patch = pylab.Rectangle(
                    (C, R), W, H, linewidth=1, edgecolor="green", facecolor="none"
                )
                axes[0].add_patch(loop_bbox_patch)
            except BaseException:
                pass

        comparison_path = prediction_img_path.replace(
            "hierarchical_mask_high_contrast_predicted", "comparison"
        )
        pylab.savefig(comparison_path)
        pylab.close()
        print("saving %s" % comparison_path)
    end = time.time()

    print(
        "%d predictions saved in %.4f seconds (%.4f per image)"
        % (len(input_images), end - _start, (end - _start) / len(input_images))
    )


def get_hierarchical_mask_from_prediction(
    prediction,
    notions=["crystal", "loop_inside", "loop", "stem", "pin", "foreground"],
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    notion_values={
        "crystal": 6,
        "loop_inside": 5,
        "loop": 4,
        "stem": 3,
        "pin": 2,
        "foreground": 1,
    },
    min_size=32,
    massage=False,
):
    hierarchical_mask = np.zeros(prediction.shape[:2])
    for notion in notions:
        notion_value = notion_values[notion]
        l = notion_indices[notion]
        mask = prediction[:, :, l] > threshold
        if massage:
            if notion in ["crystal", "loop", "loop_inside", "stem", "pin"]:
                massager = "convex"
            else:
                massager = "filled"
            mask = massage_mask(mask, min_size=min_size, massager=massager)
        if np.any(mask):
            hierarchical_mask[mask == 1] = notion_value
    return hierarchical_mask


def get_hierarchical_mask_from_kth_prediction(predictions, k):
    prediction = get_kth_prediction_from_predictions(predictions, k)
    hierarchical_mask = get_hierarchical_mask_from_prediction(prediction)
    return hierarchical_mask


def get_hierarchical_mask_from_predictions(
    predictions,
    k=0,
    notions=["crystal", "loop_inside", "loop", "stem", "pin", "foreground"],
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    notion_values={
        "crystal": 6,
        "loop_inside": 5,
        "loop": 4,
        "stem": 3,
        "pin": 2,
        "foreground": 1,
    },
    min_size=32,
    massage=False,
):
    hierarchical_mask = np.zeros(predictions[0].shape[1:3], dtype=np.uint8)
    for notion in notions[::-1]:
        notion_value = notion_values[notion]
        l = notion_indices[notion]
        mask = predictions[l][k, :, :, 0] > threshold
        if massage:
            if notion in ["crystal", "loop", "loop_inside", "stem", "pin"]:
                massager = "convex"
            else:
                massager = "filled"
            mask = massage_mask(mask, min_size=min_size, massager=massager)
        if np.any(mask):
            hierarchical_mask[mask == 1] = notion_value
    return hierarchical_mask


def get_kth_prediction_from_predictions(predictions, k):
    prediction = np.zeros(predictions.shape[1:3] + predictions.shape[0], dtype=np.uint8)
    for n, notion in enumerate(predictions):
        prediction[:, :, n] = predictions[n][k][:, :, 0]
    return prediction


def massage_mask(mask, min_size=32, massager="convex"):
    mask = remove_small_objects(mask, min_size=min_size)
    if not np.any(mask):
        return mask
    labeled_image = mask.astype("uint8")
    properties = regionprops(labeled_image)[0]
    bbox = properties.bbox
    if massager == "convex":
        mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = properties.convex_image
    elif massager == "filled":
        mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = properties.filled_image
    return mask


def get_most_likely_click_from_description(description, verbose=False):
    _start = time.time()
    gmlc = False
    most_likely_click = -1, -1
    shape = np.array(description["hierarchical_mask"].shape)
    for notion in ["crystal", "loop_inside", "loop"]:
        notion_description = description[get_notion_string(notion)]
        if notion_description["present"]:
            r = notion_description["r"]
            c = notion_description["c"]
            most_likely_click = np.array((r, c)) / shape
            if verbose:
                print("%s found!" % notion)
            gmlc = True
            break
    if gmlc is False:
        notion_description = description["foreground"]
        if notion_description["present"]:
            epo = notion_description["epo"]
            most_likely_click = np.array(epo) / shape
    if verbose:
        print("most likely click determined in %.4f seconds" % (time.time() - _start))
    return most_likely_click


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


def get_extreme_point(
    projection, pa=None, orientation="horizontal", extreme_direction=-1
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


def get_descriptions(
    predictions,
    notions=[
        "foreground",
        "crystal",
        "loop_inside",
        "loop",
        ["crystal", "loop"],
        ["crystal", "loop", "stem"],
    ],
    threshold=0.5,
    min_size=32,
    original_image_shape=(1200, 1600),
):
    descriptions = []
    foreground = get_notion_string("foreground")
    crystal_loop = get_notion_string(["crystal", "loop"])
    possible = get_notion_string(["crystal", "loop", "stem"])
    prediction_shape = np.array(predictions[0].shape[1:3])
    original_shape = np.array(original_image_shape[:2])
    for k in range(len(predictions[0])):
        description = {}
        description["original_shape"] = original_shape
        description["prediction_shape"] = prediction_shape
        description["hierarchical_mask"] = get_hierarchical_mask_from_predictions(
            predictions, k=k
        )
        for notion in notions:
            (
                present,
                r,
                c,
                h,
                w,
                r_max,
                c_max,
                r_min,
                c_min,
                bbox,
                area,
                notion_mask,
            ) = get_notion_prediction(predictions, notion, k=k, min_size=min_size)
            if present:
                epo, epi, epooa, epioa, pa = get_extreme_point(notion_mask)
            else:
                epo, epi, epooa, epioa = 4 * [(-1, -1)]
                pa = np.nan
            # description[get_notion_string(notion)] = (present, (r, c, h, w, area), (epo, epi, epooa, epioa, pa), notion_mask)
            description[get_notion_string(notion)] = {
                "present": present,
                "r": r,
                "c": c,
                "h": h,
                "w": w,
                "area": area,
                "epo": epo,
                "epi": epi,
                "epooa": epooa,
                "epioa": epioa,
                "pa": pa,
                "notion_mask": notion_mask,
            }

        epo_cil, epi_cil, epooa_cil, epioa_cil, pa_cil = get_extreme_point(
            description[crystal_loop]["notion_mask"], pa=description[possible]["pa"]
        )
        description["present"] = description[foreground]["present"]
        description["most_likely_click"] = get_most_likely_click_from_description(
            description
        )
        description["aoi_bbox"] = get_bbox_from_description(
            description, notions=[["crystal", "loop"], "foreground"]
        )
        description["crystal_bbox"] = get_bbox_from_description(
            description, notions=["crystal"]
        )
        description["extreme"] = description[foreground]["epo"] / prediction_shape
        description["end_likely"] = (
            description[crystal_loop]["epooa"] / prediction_shape
        )
        description["start_likely"] = (
            epioa_cil / prediction_shape
        )  # description[crystal_loop]['epioa']
        description["start_possible"] = (
            description[possible]["epioa"] / prediction_shape
        )
        descriptions.append(description)
    return descriptions


def get_predictions(request_arguments, host="localhost", port=89019, verbose=False):
    start = time.time()
    context = zmq.Context()
    if verbose:
        print("Connecting to server ...")
    socket = context.socket(zmq.REQ)
    socket.connect("tcp://%s:%d" % (host, port))
    socket.send(pickle.dumps(request_arguments))
    raw_predictions = socket.recv()
    predictions = pickle.loads(raw_predictions)
    if verbose:
        print("Received predictions in %.4f seconds" % (time.time() - start))
    return predictions


def efficient_resize(img, new_size, anti_aliasing=True):
    return (
        img_to_array(array_to_img(img).resize(new_size[::-1]), dtype="float32") / 255.0
    )


def get_notion_description(mask, min_size=32):
    present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, properties = [
        np.nan
    ] * 12
    present = 0

    if np.any(mask):
        labeled_image = mask.astype("uint8")
        properties = regionprops(labeled_image)[0]

        if properties.convex_area > min_size:
            present = 1
            area = properties.convex_area
        else:
            present = 0
        bbox = properties.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        r, c = properties.centroid
        c_max = bbox[3]
        r_max = ndi.center_of_mass(labeled_image[:, c_max - 5 : c_max])[0]
        c_min = bbox[1]
        r_min = ndi.center_of_mass(labeled_image[:, c_min : c_min + 5])[0]

    return present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, properties


def get_notion_mask_from_predictions(
    predictions,
    notion,
    k=0,
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    min_size=32,
):
    notion_mask = np.zeros(predictions[0].shape[1:3], dtype=bool)

    if isinstance(notion, list):
        for n in notion:
            index = notion_indices[n]
            noti_pred = predictions[index][k, :, :, 0] > threshold
            noti_pred = remove_small_objects(noti_pred, min_size=min_size)
            notion_mask = np.logical_or(notion_mask, noti_pred)

    elif isinstance(notion, str):
        index = notion_indices[notion]
        notion_mask = predictions[index][k, :, :, 0] > threshold
        notion_mask = remove_small_objects(notion_mask, min_size=min_size)
    return notion_mask


def get_notion_mask_from_masks(
    masks,
    notion,
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    min_size=32,
):
    notion_mask = np.zeros(masks.shape[:2], dtype=bool)

    if isinstance(notion, list):
        for n in notion:
            index = notion_indices[n]
            noti_mask = masks[:, :, index]
            noti_mask = remove_small_objects(noti_mask > 0, min_size=min_size)
            notion_mask = np.logical_or(notion_mask, noti_mask)
    elif isinstance(notion, str):
        index = notion_indices[notion]
        notion_mask = masks[:, :, index]
        notion_mask = remove_small_objects(notion_mask > 0, min_size=min_size)
    return notion_mask


def get_notion_prediction(
    predictions,
    notion,
    k=0,
    notion_indices={
        "crystal": 0,
        "loop_inside": 1,
        "loop": 2,
        "stem": 3,
        "pin": 4,
        "foreground": 5,
    },
    threshold=0.5,
    min_size=32,
):
    if isinstance(predictions, list):
        notion_mask = get_notion_mask_from_predictions(
            predictions,
            notion,
            k=k,
            notion_indices=notion_indices,
            threshold=threshold,
            min_size=min_size,
        )
    elif isinstance(predictions, np.ndarray) and len(predictions.shape) == 3:
        notion_mask = get_notion_mask_from_masks(
            predictions, notion, notion_indices=notion_indices, min_size=min_size
        )

    (
        present,
        r,
        c,
        h,
        w,
        r_max,
        c_max,
        r_min,
        c_min,
        bbox,
        area,
        properties,
    ) = get_notion_description(notion_mask, min_size=min_size)

    if not isinstance(properties, float):
        if notion == "foreground" or isinstance(notion, list):
            notion_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = properties.filled_image
        else:
            notion_mask[bbox[0] : bbox[2], bbox[1] : bbox[3]] = properties.convex_image

    return (
        present,
        r,
        c,
        h,
        w,
        r_max,
        c_max,
        r_min,
        c_min,
        bbox,
        area,
        notion_mask.astype("uint8"),
    )


def get_most_likely_click(predictions, k=0, verbose=False, min_size=32):
    _start = time.time()
    gmlc = False
    most_likely_click = -1, -1

    for notion in ["crystal", "loop_inside", "loop"]:
        notion_prediction = get_notion_prediction(
            predictions, notion, k=k, min_size=min_size
        )
        if notion_prediction[0] == 1:
            most_likely_click = (
                notion_prediction[1] / notion_prediction[-1].shape[0],
                notion_prediction[2] / notion_prediction[-1].shape[1],
            )
            if verbose:
                print("%s found!" % notion)
            gmlc = True
            break
    if gmlc is False:
        foreground = get_notion_prediction(
            predictions, "foreground", k=k, min_size=min_size
        )
        if foreground[0] == 1:
            most_likely_click = (
                foreground[5] / foreground[-1].shape[0],
                foreground[6] / foreground[-1].shape[1],
            )
    if verbose:
        print("most likely click determined in %.4f seconds" % (time.time() - _start))
    return most_likely_click


def get_bbox_from_description(description, notions=[["crystal", "loop"], "foreground"]):
    shape = description["hierarchical_mask"].shape
    for notion in notions:
        notion_description = description[get_notion_string(["crystal", "loop"])]
        present = notion_description["present"]
        if present:
            r = notion_description["r"]
            c = notion_description["c"]
            h = notion_description["h"]
            w = notion_description["w"]
            r /= shape[0]
            c /= shape[1]
            h /= shape[0]
            w /= shape[1]
            break
        else:
            r, c, h, w = 4 * [np.nan]
    return present, r, c, h, w


def get_notion_string(notion):
    if isinstance(notion, list):
        notion_string = ",".join(notion)
    else:
        notion_string = notion
    return notion_string


def get_loop_bbox(predictions, k=0, min_size=32):
    (
        loop_present,
        r,
        c,
        h,
        w,
        r_max,
        c_max,
        r_min,
        c_min,
        bbox,
        area,
        notion_prediction,
    ) = get_notion_prediction(
        predictions, ["crystal", "loop_inside", "loop"], k=k, min_size=min_size
    )
    shape = predictions[0].shape[1:3]
    if bbox is not np.nan:
        r = bbox[0] + h / 2
        c = bbox[1] + w / 2
    r /= shape[0]
    c /= shape[1]
    h /= shape[0]
    w /= shape[1]
    return loop_present, r, c, h, w


def get_raw_projections(predictions, notion="foreground", threshold=0.5, min_size=32):
    raw_projections = []
    for k in range(len(predictions[0])):
        (
            present,
            r,
            c,
            h,
            w,
            r_max,
            c_max,
            r_min,
            c_min,
            bbox,
            area,
            notion_mask,
        ) = get_notion_prediction(predictions, notion, k=k, min_size=min_size)
        raw_projections.append((present, (r, c, h, w), notion_mask))
    return raw_projections


def segment(base="/nfs/data2/Martin/Research/murko", epochs=25, patience=5):
    keras.mixed_precision.set_global_policy("mixed_float16")
    date = "start"
    date_next = "2021-12-13_tiramisu_SeparableConv2D"
    model_and_weights = os.path.join(base, "sample_segmentation_%s.h5" % date)
    next_model_and_weights = os.path.join(
        base, "sample__segmentation_%s.h5" % date_next
    )
    model_train_history = os.path.join(
        base, "sample_segmentation_%s_history.pickle" % date
    )
    # model = get_other_model(model_img_size, num_classes)
    # Split our img paths into a training and a validation set
    # train_paths, train_target_img_paths = get_training_dataset()
    # val_paths, val_target_img_paths = get_validation_dataset()
    train_paths, val_paths = get_training_and_validation_datasets()
    print(
        "training on %d samples, validating on %d samples"
        % (len(train_paths), len(val_paths))
    )

    # Instantiate data Sequences for each split
    train_gen = SampleSegmentationDataset(
        batch_size, model_img_size, train_paths, augment=True
    )
    val_gen = SampleSegmentationDataset(
        batch_size, model_img_size, val_paths, augment=False
    )

    model = get_tiramisu(convolution_type="SeparableConv2D")

    # new_date = '2021-11-12_zoom123'
    checkpointer = keras.callbacks.ModelCheckpoint(
        next_model_and_weights, verbose=1, mode="min", save_best_only=True
    )
    earlystopper = keras.callbacks.EarlyStopping(patience=patience, verbose=1)
    callbacks = [checkpointer, earlystopper]

    # Train the model, doing validation at the end of each epoch.
    # strategy = tf.distribute.MirroredStrategy()
    # with strategy.scope():
    # model = get_tiramisu(convolution_type='SeparableConv2D')
    # if os.path.isfile(model_and_weights):
    # model.load_weights(model_and_weights)
    # history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    history = model.fit(
        train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks
    )
    f = open(model_train_history, "wb")
    pickle.dump(history.history, f)
    f.close()
    # Generate predictions for all images in the validation set
    # val_gen = SampleSegmentationDataset(batch_size, img_size, val_paths, val_target_img_paths)
    _start = time.time()
    val_preds = model.predict(val_gen)
    print(
        "predicting %d examples took %.4f seconds"
        % (len(val_preds), time.time() - _start)
    )
    epochs = range(1, len(history.history["loss"]) + 1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    pylab.figure()
    pylab.plot(epochs, loss, "bo-", label="Training loss")
    pylab.plot(epochs, val_loss, "ro-", label="Validation loss")
    pylab.title("Training and validation loss")
    pylab.legend()
    pylab.savefig("sample_foreground_segmentation_%s.png" % date)
    pylab.show()


def get_cpi_from_user_click(
    user_click,
    img_size,
    resize_factor,
    img_path,
    click_radius=320e-3,
    zoom=1,
    scale_click=False,
):
    if all(np.array(user_click) >= 0):
        try:
            _y = int(user_click[0])
            _x = int(min(user_click[1], img_size[1]))
            if all(np.array((_y, _x)) >= 0):
                cpi = click_probability_image(
                    _x,
                    _y,
                    img_size,
                    click_radius=click_radius,
                    zoom=zoom,
                    resize_factor=resize_factor,
                    scale_click=scale_click,
                )
            else:
                cpi = np.zeros(img_size, dtype="float32")
        except BaseException:
            print(traceback.print_exc())
            os.system("echo %s >> click_generation_problems_new.txt" % img_path)
            return None
    else:
        cpi = np.zeros(img_size, dtype="float32")
    cpi = np.expand_dims(cpi, axis=2)
    return cpi


def get_resize_and_rescale(model_img_size):
    resize_and_rescale = keras.Sequential(
        [
            # layers.Resizing(model_img_size, model_img_size),
            layers.Rescaling(1.0 / 255)
        ]
    )
    return resize_and_rescale


def analyse_histories(
    notions=["crystal", "loop_inside", "loop", "stem", "pin", "foreground"]
):
    histories = (
        glob.glob("*.history")
        + glob.glob("experiments/*.history")
        + glob.glob("backup/*.history")
    )
    metrics_table = {}
    for history in histories:
        print(history)
        h = pickle.load(open(history, "rb"), encoding="bytes")
        plot_history(history, h=h, notions=notions)
        val_metrics = []
        for notion in notions:
            key = "val_%s_BIoU_1" % notion
            if key in h:
                val_metrics.append(h["val_%s_BIoU_1" % notion])
        val_metrics = np.array(val_metrics)
        try:
            best = val_metrics.max(axis=1).T
            best
        except BaseException:
            best = "problem in determining expected metrics"

        line = "%s: %s" % (best, history)
        print(line)
        os.system('echo "%s" >> histories.txt' % line)


def resize_images(images, size, method="bilinear", align_corners=False):
    """See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        "bilinear": tf.image.ResizeMethod.BILINEAR,
        "nearest": tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        "bicubic": tf.image.ResizeMethod.BICUBIC,
        "area": tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, methods[method], align_corners)


def compare(h1, h2, what="crystal"):
    pylab.figure(1)
    for key in h1:
        if what in key and "loss" not in key:
            pylab.plot(h1[key], label=key)
    pylab.legend()
    pylab.figure(2)
    for key in h2:
        if what in key and "loss" not in key:
            pylab.plot(h2[key], label=key)
    pylab.legend()
    pylab.show()


def get_pixels(
    directory="/nfs/data2/Martin/Research/murko/images_and_labels",
    notions=[
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "capillary",
        "ice",
        "foreground",
    ],
    print_table=True,
):
    masks = glob.glob("%s/*/masks.npy" % directory)
    pixel_counts = dict([(notion, 0) for notion in notions])
    pixel_counts["total"] = 0
    for mask in masks:
        m = np.load(mask)
        for k, notion in enumerate(notions):
            pixel_counts[notion] += m[:, :, k].sum()
        pixel_counts["total"] += np.prod(m.shape[:2])
    if print_table:
        print(
            "total pixels %d (%.3fG)".ljust(15)
            % (pixel_counts["total"], pixel_counts["total"] / 1e9)
        )
        print(
            "total foreground %d (%.3fG, %.4f of all)".ljust(15)
            % (
                pixel_counts["foreground"],
                pixel_counts["foreground"] / 1e9,
                pixel_counts["foreground"] / pixel_counts["total"],
            )
        )
        print()
        print(
            "notion".rjust(15),
            "fraction_label".rjust(15),
            "fraction_total".rjust(15),
            "weight_label".rjust(20),
            "weight_total".rjust(20),
        )
        for key in pixel_counts:
            print(
                key.rjust(15),
                "%.4f".rjust(10) % (pixel_counts[key] / pixel_counts["foreground"]),
                "%.4f".rjust(15) % (pixel_counts[key] / pixel_counts["total"]),
                "%3.1f".zfill(2).rjust(20)
                % (pixel_counts["foreground"] / pixel_counts[key]),
                "%3.1f".zfill(2).rjust(20)
                % (pixel_counts["total"] / pixel_counts[key]),
            )
    return pixel_counts


def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()

    with graph.as_default():
        with session.as_default():
            model = keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()

            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(
                graph=graph, run_meta=run_meta, cmd="op", options=opts
            )

            return flops.total_float_ops
