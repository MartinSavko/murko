#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

import os
import json
import cv2 as cv
import time
import simplejpeg
import tifffile
import glob
import numpy as np
from labelme import utils
import traceback
import pickle
from pprint import pprint

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


def get_labelme_shapes_from_ade_annotation(annotation):
    shapes = []
    for item in annotation["object"]:
        shape = {}
        shape["label"] = item["name"]
        shape["group_id"] = None
        shape["description"] = ""
        shape["flags"] = {}
        shape["mask"] = None
        points = list(zip(item["polygon"]["x"], item["polygon"]["y"]))
        if len(points) >= 3:
            shape["shape_type"] = "polygon"
            shape["points"] = points
            shapes.append(shape)
        else:
            print("could not handle this one, please check")
            print(item)
            print(annotation)
    return shapes
    
def transform_ADE_dataset(source="/nfs/data4/Martin/Research/ADE20K_2021_17_01/index_ade20k.pkl", destination="json", to_remove="ADE20K_2021_17_01/images/ADE"):
    
    _start = time.time()
    ade = pickle.load(open(source, "rb"))
    destination = os.path.join(os.path.dirname(source), "json")
    print(destination)
    for folder, filename in zip(ade["folder"], ade["filename"]):
        image_path = os.path.join(os.path.dirname(os.path.dirname(source)), folder, filename)
        annotation_path = image_path.replace(".jpg", ".json")
        annotation = json.load(open(annotation_path))["annotation"]
        
        img = cv.imread(image_path)
        image_data = utils.img_arr_to_b64(img)
        image_height, image_width, channels = annotation["imsize"] #img.shape
        
        shapes = get_labelme_shapes_from_ade_annotation(annotation)
        jf = get_njf = get_new_json_file(
            shapes,
            image_path,
            imageHeight=image_height,
            imageWidth=image_width,
            imageData=image_data,
        )

        fname = os.path.join(destination, folder[len(to_remove)+1:], filename.replace(".jpg", ".json"))
        save_json_file(jf, fname)
        
    print(f"ADE20k's {len(ade['filename'])} samples transformed in {time.time() - _start0:.3f} seconds")

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
        image_path = os.path.join(destination, image["file_name"])
        #image_data = simplejpeg.encode_jpeg(cv.imread(image_path))
        image_data = utils.img_arr_to_b64(cv.imread(image_path))
        image_width = image["width"]
        image_height = image["height"]
        
        jf = get_new_json_file(
            shapes[image["id"]],
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

def transform_cristal_dataset(source='/nfs/data2/Martin/Research/hubert', destination="json"):  
    
    _start = time.time()
    
    tifs = glob.glob(os.path.join(source, "tifs/*.tif"))

    for image_path in tifs:
        template = image_path.replace(".tif", "")
        img = tifffile.imread(image_path)
        
        bit_depth = 8
        if img.max() > 255:
            bit_depth = np.ceil(np.log2(img.max()))
        
        img = (img / (2**bit_depth - 1)) 
        img *= 255

        imgc = np.zeros(img.shape + (3, ), dtype=np.uint8)
        imgc = img.astype(np.uint8)
        
        image_data = utils.img_arr_to_b64(imgc)
        
        _mask = glob.glob(os.path.join(source, "masks", f"{os.path.basename(template)}*"))
        if len(_mask):
            if _mask[0].endswith(".png"):
                mask = cv.imread(_mask[0])
            elif (_mask[0].endswith(".tiff") or _mask[0].endswith(".tif")):
                mask = tifffile.imread(_mask[0])
        else:
            print(f"problem {template} {_mask}")

        if len(mask.shape) > 2:
            mask = mask.mean(axis=2)
        mask = mask == mask.max()
        mask = mask.astype(np.uint8)
        
        try:
            shape = get_labelme_shape_from_mask(mask, "capillary")
            image_height, image_width = img.shape
            
            jf = get_new_json_file(
                [shape],
                image_path,
                imageHeight=image_height,
                imageWidth=image_width,
                imageData=image_data,
            )
            
            save_json_file(jf, os.path.join(source, destination, f"{os.path.basename(template)}.json"))
        except:
            print(f"mask {mask.min()}, {mask.max()}")
            print(f" {mask.sum()}")
            print(f"problem in {image_path}")
            
            print(_mask)
            traceback.print_exc()
            

    print(f"{len(tifs)} examples in cristal dataset transformed in {time.time() - _start:.3f} seconds")
    
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

    #transform_cristal_dataset()
    #transform_pcs_dataset(args.source)
    transform_ADE_dataset()
    
