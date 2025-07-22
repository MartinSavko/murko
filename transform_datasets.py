#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import os
import json
import cv2 as cv
import time
import simplejpeg
from labelme import utils


def get_new_json_file(
    shapes,  # list of shape directories
    imagePath,
    imageHeight=None,
    imageWidth=None,
    version="5.4.1",
    imageData=None,
    flags={},
):
    json_file = {}
    json_file["version"] = version
    json_file["flags"] = flags
    json_file["shapes"] = shapes
    json_file["imagePath"] = imagePath
    if imageHeight is None or imageWidth is None:
        img = imageio.imread(imagePath)
        imageHeight, imageWidth = img.shape[:2]
    json_file["imageHeight"] = imageHeight
    json_file["imageWidth"] = imageWidth
    json_file["imageData"] = imageData

    return json_file


def transform_pcs_dataset(pcs="pcs_validation.json", output="_labelme"):
    _start0 = time.time()
    original = json.load(open(pcs, "r"))
    print(
        f'original records loaded in {time.time() - _start0:.3f} seconds'
    )
    shapes = {}
    _start1 = time.time()
    for annotation in original["annotations"]:
        image_id = annotation["image_id"]
        shape = get_labelme_shape_from_pcs_annotation(annotation)
        if image_id in shapes:
            shapes[image_id].append(shape)
        else:
            shapes[image_id] = [shape]

    print(
        f'{len(original["annotations"])} annotations processed in {time.time() - _start1:.3f} seconds'
    )

    destination = pcs.replace(".json", "") + output
    
    _start2 = time.time()
    for image in original["images"]:
        image_path = os.path.join(pcs.replace(".json", ""), image["file_name"])
        #image_data = simplejpeg.encode_jpeg(cv.imread(image_path))
        image_data = utils.img_arr_to_b64(cv.imread(image_path))
        image_width = image["width"]
        image_height = image["height"]
        image_id = image["id"]
        jf = get_new_json_file(
            shapes[image_id],
            image_path,
            imageHeight=image_height,
            imageWidth=image_width,
            imageData=image_data,
        )

        save_json_file(jf, os.path.join(destination, image_path.replace(".png", ".json")))

    print(
        f'{len(original["images"])} images processed in {time.time() - _start2:.3f} seconds'
    )
    print(f"{pcs} transformed in {time.time() - _start0:.3f} seconds")


def get_labelme_shape_from_pcs_annotation(annotation, categories={1: "crystal"}):
    shape = {}
    shape["label"] = categories[annotation["category_id"]]
    shape["group_id"] = None
    shape["description"] = ""
    shape["flags"] = {}
    shape["mask"] = None
    s = annotation["segmentation"][0]
    points = [[s[2 * k], s[2 * k + 1]] for k in range(int(len(s) / 2))]
    assert len(points) >= 3
    shape["shape_type"] = "polygon"
    shape["points"] = points
    return shape


def get_labelme_shape_from_mask(mask, label):
    shape = {}
    shape["label"] = label
    shape["group_id"] = None
    shape["description"] = ""
    shape["flags"] = {}
    shape["mask"] = None
    contours = cv.findContours(
        mask.astype(np.uint8), cv.RETR_EXTERNAL, cv.CHAIN_APPROX_SIMPLE
    )
    points = contours[0][0].astype(float)
    points = points.reshape((points.shape[0], points.shape[-1]))
    # points = points[:, ::-1]
    pts = []
    for p in points:
        pts.append(list(p))
    if len(points) >= 3:
        shape_type = "polygon"
    elif len(points) == 2:
        shape_type = "rectangle"
    else:
        shape_type = "point"
    shape["shape_type"] = shape_type
    shape["points"] = pts
    return shape


def get_labelme_shapes_from_chimp_record(imagepath):
    realpath = os.path.realpath(imagepath)
    record = np.load(realpath.replace("images", "masks").replace(".jpg", ".npz"))
    labels = record["class_labels"]
    masks = record["masks"]
    shapes = []
    for mask, label in zip(masks, labels):
        shape = get_labelme_shape_from_mask(mask, label)
        shapes.append(shape)

    return shapes


def create_labelme_file_from_chimp_record(imagepath):
    realpath = os.path.realpath(imagepath)
    shapes = get_labelme_shapes_from_chimp_record(realpath)
    jsonpath = realpath.replace("images", "json").replace(".jpg", ".json")

    json_file = get_new_json_file(shapes, realpath)

    save_json_file(json_file, jsonpath)


def save_json_file(content, path):
    if not os.path.isdir(os.path.dirname(path)):
        os.makedirs(os.path.dirname(path))
    fp = open(path, "w")
    json.dump(content, fp)
    fp.close()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-s", "--source", default="pcs_validation.json", type=str, help="source"
    )

    args = parser.parse_args()

    transform_pcs_dataset(args.source)
