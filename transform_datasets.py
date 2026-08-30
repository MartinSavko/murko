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
import imageio

from regionprops import get_mask_boundary

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
    could_not_handle = False
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
            could_not_handle = True
    if could_not_handle:
        print("There were unhandled objects in this annotation")
        print(annotation)
        print()
    return shapes
    
def transform_ADE_dataset(source="/nfs/data4/Martin/Research/ADE20K_2021_17_01/index_ade20k.pkl", destination="json", to_remove="ADE20K_2021_17_01/images/ADE"):
    
    _start = time.time()
    ade = pickle.load(open(source, "rb"))
    destination = os.path.join(os.path.dirname(source), "json")
    print(destination)
    skipped = 0
    failed = 0
    for folder, filename in zip(ade["folder"], ade["filename"]):
        destination_fname = os.path.join(destination, folder[len(to_remove)+1:], filename.replace(".jpg", ".json"))
        if os.path.isfile(destination_fname): 
            skipped += 1
            continue
    
        image_path = os.path.join(os.path.dirname(os.path.dirname(source)), folder, filename)
        annotation_path = image_path.replace(".jpg", ".json")
        try:
            annotation = json.load(open(annotation_path))["annotation"]
        except UnicodeDecodeError:
            annotation = json.load(open(annotation_path, 'rb'))["annotation"]
        except:
            failed += 1
            print(f"Could not load {annotation_path}, please check")
            traceback.print_exc()
            print()
            continue
        
        img = cv.imread(image_path)
        
        if len(img.shape) == 3:
            image_height, image_width, channels = img.shape
            image_data = utils.img_arr_to_b64(img)
        elif len(img.shape) == 2:
            image_height, image_width = img.shape
            imgc = np.zeros(img.shape + (3, ), dtype=np.uint8)
            imgc = img
            image_data = utils.img_arr_to_b64(img)
        else:
            print(f"Could not handle {annotation_path}, please check!")
            continue
        
        shapes = get_labelme_shapes_from_ade_annotation(annotation)
        jf = get_new_json_file(
            shapes,
            image_path,
            imageHeight=image_height,
            imageWidth=image_width,
            imageData=image_data,
        )

        save_json_file(jf, destination_fname)
        
    print(f"ADE20k's {len(ade['filename'])} ({skipped} skipped, {failed} failed) samples transformed in {time.time() - _start0:.3f} seconds")

def transform_coco_style_dataset(source="validation/labels.json", destination="labelme/validation", include_image_data=False):
    _start0 = time.time()
    
    original = json.load(open(source, "r"))
    
    categories = {}
    for item in original["categories"]:
        categories[item["id"]] = item["name"]

    print(
        f'original records loaded in {time.time() - _start0:.3f} seconds'
    )
    
    destination = os.path.realpath(destination)
    
    shapes = {}
    _start1 = time.time()
    for annotation in original["annotations"]:
        image_id = annotation["image_id"]
        if type(annotation["segmentation"]) is dict:
            continue
        shape = get_labelme_shape_from_coco_style_annotation(annotation, categories=categories)
        if image_id in shapes:
            shapes[image_id].append(shape)
        else:
            shapes[image_id] = [shape]

    print(
        f'{len(original["annotations"])} annotations processed in {time.time() - _start1:.3f} seconds'
    )

    _start2 = time.time()
    for image in original["images"]:
        image_path = os.path.join(os.path.realpath(os.path.dirname(source)), "data", image["file_name"])
        #image_data = simplejpeg.encode_jpeg(cv.imread(image_path))
        if include_image_data:
            image_data = utils.img_arr_to_b64(cv.imread(image_path))
        else:
            image_data = None
        image_width = image["width"]
        image_height = image["height"]
        image_id = image["id"]
        if image_id in shapes:
            jf = get_new_json_file(
                shapes[image_id],
                image_path,
                imageHeight=image_height,
                imageWidth=image_width,
                imageData=image_data,
            )

            save_json_file(jf, os.path.join(destination, f"{image_id:07d}.json"))

    print(
        f'{len(original["images"])} images processed in {time.time() - _start2:.3f} seconds'
    )
    print(f"{source} transformed in {time.time() - _start0:.3f} seconds")


def get_labelme_shape_from_coco_style_annotation(annotation, categories={1: "crystal"}):
    shape = {}
    try:
        s = annotation["segmentation"][0]
    except:
        print("problem with getting segmentation out, please check")
        print(annotation)
        return shape
    
    shape["label"] = categories[annotation["category_id"]]
    shape["group_id"] = None
    shape["description"] = ""
    shape["flags"] = {}
    shape["mask"] = None
        
    points = [[s[2 * k], s[2 * k + 1]] for k in range(int(len(s) / 2))]
    assert len(points) >= 3
    shape["shape_type"] = "polygon"
    shape["points"] = points
    return shape

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
    

def get_labelme_shape_from_mask(mask, label):
    shape = {}
    shape["label"] = label
    shape["group_id"] = None
    shape["description"] = ""
    shape["flags"] = {}
    shape["mask"] = None

    points = get_mask_boundary(mask, approximate=True)
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

def get_empty_shape(label=None, points=[], shape_type=""):
    shape = {}
    shape["label"] = label
    shape["group_id"] = None
    shape["description"] = ""
    shape["flags"] = {}
    shape["mask"] = None
    shape["shape_type"] = shape_type
    shape["points"] = points
    return shape

def generate_json_files_for_backgrounds(
    directory="/nfs/data2/Martin/Research/murko/manually_segmented_images/json/backgrounds",
    template="*.jpg",
    unit_square=np.array([[0, 0], [0, 1], [1, 1], [1, 0]]),
):
    bfs = glob.glob(os.path.join(directory, template))
    print("background files")
    print(bfs)
    for bf in bfs:
        print(bf)
        image = imageio.imread(bf)
        image_shape = np.array(image.shape[:2])
        polygon = image_shape[::-1] * unit_square
        points = [[int(item[0]), int(item[1])] for item in polygon]
        shape = get_empty_shape(label="background", points=points, shape_type="polygon")

        json_path = bf.replace(".jpg", ".json")
        json_file = get_new_json_file(
            [shape],
            os.path.basename(bf),
            imageWidth=int(image_shape[1]),
            imageHeight=int(image_shape[0]),

        )
        print("json_file")
        print(json_file)
        save_json_file(json_file, json_path)

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    

    parser.add_argument(
        "-s", "--source", default="pcs_validation.json", type=str, help="source"
    )
    
    parser.add_argument(
        "-d", "--destination", default="labelme/validation", type=str, help="destination"
    )
    
    parser.add_argument(
        "--include_image_data", action="store_true", help="include image data"
    )
    
    args = parser.parse_args()

    #transform_cristal_dataset()
    #transform_pcs_dataset(args.source)
    #transform_ADE_dataset()
    transform_coco_style_dataset(args.source, args.destination, include_image_data=bool(args.include_image_data))
    
