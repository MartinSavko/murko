#!/usr/bin/python3
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import simplejpeg
import PIL.Image
import argparse
import sys
import base64
import json
import os
import os.path as osp
import numpy as np
import pylab
from tensorflow.keras.preprocessing.image import (
    save_img,
    load_img,
    img_to_array,
    array_to_img,
)

# sys.path.insert(0, '/usr/local/lib/python3.8/dist-packages')
from labelme import utils
from labelme.logger import logger
from labelme.utils import shape_to_mask
from skimage.morphology.footprints import disk
import time
import warnings

warnings.filterwarnings("ignore")


def generate_background_masks(
    imagepath,
    destination="images_and_labels",
    labels="not_background,pin,stem,loop,loop_inside,ice,dust,capillary,crystal",
):
    (h, w, colspc, sbsmpling) = simplejpeg.decode_jpeg_header(
        open(imagepath, "rb").read()
    )
    n_categories = len(labels.split(","))

    segmentation_mask = np.zeros((h, w, n_categories), dtype=int)
    user_click = np.array([-1, -1])

    output_directory = os.path.join(
        destination, os.path.basename(imagepath).replace(".jpg", "")
    )
    if not os.path.isdir(output_directory):
        os.makedirs(output_directory)
    np.save(os.path.join(output_directory, "masks.npy"), segmentation_mask)
    np.save(os.path.join(output_directory, "masks.npy"), user_click)


def convert(
    json_file,
    out=None,
    labels="not_background,pin,stem,loop,loop_inside,ice,dust,capillary,crystal",
    additional_labels="cd_loop,cd_stem,drop",
    additional_labels_mapping={
        "cd_loop": "loop",
        "cd_stem": "stem",
        "drop": "not_background",
    },
):
    data = json.load(open(json_file))

    imageData = data.get("imageData")

    if not imageData:
        imagePath = os.path.join(os.path.dirname(json_file), data["imagePath"])
        with open(imagePath, "rb") as f:
            imageData = f.read()
            imageData = base64.b64encode(imageData).decode("utf-8")
    img = utils.img_b64_to_arr(imageData)

    if out is None:
        out_dir = osp.basename(json_file).replace(".", "_")
        out_dir = osp.join(osp.dirname(json_file), out_dir)
    else:
        out_dir = out
    if not osp.exists(out_dir):
        os.mkdir(out_dir)

    user_click = [-1, -1]
    for item in data["shapes"]:
        if item["label"] == "user_click" and item["shape_type"] == "point":
            user_click = item["points"][0][::-1]

    _shapes = []
    for label in labels.split(","):
        for item in data["shapes"]:
            if item["label"] == label:
                _shapes.append(item)

    for label in additional_labels.split(","):
        for item in data["shapes"]:
            if item["label"] == label:
                item["label"] = additional_labels_mapping[label]
                _shapes.append(item)

    image_shape = (data["imageHeight"], data["imageWidth"])
    masks = {}
    for shape in _shapes:
        mask = shape_to_mask(image_shape, shape["points"])
        if shape["label"] not in masks:
            masks[shape["label"]] = mask
        else:
            masks[shape["label"]] = np.logical_or(masks[shape["label"]], mask)

    notions = [
        "crystal",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "capillary",
        "ice",
        "foreground",
    ]
    notion_masks = dict(
        [(notion, np.zeros(image_shape, dtype=np.uint8)) for notion in notions]
    )

    for item in masks:
        notion_masks["foreground"] = np.logical_or(
            notion_masks["foreground"], masks[item]
        )
    notion_masks["foreground"] = notion_masks["foreground"].astype(dtype=np.uint8)

    for notion in notions[:-1]:
        if notion in masks:
            notion_masks[notion] = masks[notion].astype(dtype=np.uint8)
        else:
            notion_masks[notion] = np.zeros(image_shape, dtype=np.uint8)

    # handle stem/pin boundary
    pin_stem_intersection = np.logical_and(notion_masks["pin"], notion_masks["stem"])
    if np.any(pin_stem_intersection):
        notion_masks["stem"][pin_stem_intersection] = 0
    # handle loop/stem boundary
    stem_loop_intersection = np.logical_and(notion_masks["loop"], notion_masks["stem"])
    if np.any(stem_loop_intersection):
        notion_masks["loop"][stem_loop_intersection] = 0

    hierarchical_mask = np.zeros(image_shape, dtype=np.uint8)
    all_masks = np.zeros(image_shape + (len(notions),), dtype=np.uint8)
    for notion in notions[::-1]:
        mask = notion_masks[notion]
        all_masks[:, :, notions.index(notion)] = mask
        if np.any(mask):
            hierarchical_mask[mask == 1] = notions.index(notion) + 1

    all_masks = np.array(all_masks)
    print("all_masks", all_masks.shape, all_masks.dtype)
    PIL.Image.fromarray(img).save(osp.join(out_dir, "img.jpg"))
    save_img(osp.join(out_dir, "imgk.jpg"), array_to_img(img), scale=False)
    save_img(osp.join(out_dir, "imgj.jpg"), array_to_img(img), scale=True)
    for notion in notions:
        nm = np.expand_dims(notion_masks[notion], 2)
        save_img(osp.join(out_dir, "%s.png" % notion), nm, scale=False)
        save_img(osp.join(out_dir, "%s_high_contrast.png" % notion), nm, scale=True)
    hm = np.expand_dims(hierarchical_mask, 2)
    print("hm", hm.shape, hm.min(), hm.max(), hm.dtype)
    save_img(osp.join(out_dir, "hierarchical_mask.png"), hm, scale=False)
    y, x = map(int, user_click)
    try:
        d = disk(5)
        d *= len(notions)
        hm[y - 5 : y + 6, x - 5 : x + 6, 0] = d
    except BaseException:
        import traceback

        traceback.print_exc()

    save_img(osp.join(out_dir, "hierarchical_mask_high_contrast.png"), hm, scale=True)
    np.save(osp.join(out_dir, "masks.npy"), all_masks)
    np.save(osp.join(out_dir, "user_click.npy"), user_click)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("json_file")
    parser.add_argument("-o", "--out", default=None)
    parser.add_argument(
        "-l",
        "--labels",
        type=str,
        default="not_background,pin,stem,loop,loop_inside,ice,dust,capillary,crystal",
    )
    parser.add_argument(
        "-a", "--additional_labels", type=str, default="cd_loop,cd_stem,drop"
    )
    args = parser.parse_args()

    print(args)

    _start = time.time()
    convert(
        args.json_file,
        out=args.out,
        labels=args.labels,
        additional_labels=args.additional_labels,
    )
    _end = time.time()
    print("time taken %.4f seconds" % (_end - _start))
