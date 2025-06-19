#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy

a = """How to represent and learn points ?
I see two options: 

a) learn offset(s) from point for each learning location i.e. have two location maps, one for horizontal offset and one for vertical offset. 
    1.) have separate head for each of the point categories. i.e. "loop_inner_point" would be learned separately from "crystal_inner_point". ( But what if we have more then one crystal in the image?)
    2.) each category of points has one learing output i.e. "inner_centers"
    each learning location than learns offset to the nearest inner point.

b) learn 2d representation of point locations, 
    1.) a supperposition of gaussians for "inner_centers", "centroids", "bbox_centers". 
        What about "extreme_points", "eigen_points" and "global_keypoints"? *) Each subcategory e.g. "extreme_point_topmost" or "extreme" or "start_likely" has a separate output.
        **) "extreme_points" has one output (four peak gaussian mixture)
            "start_likely" has one output (one gaussian)
    2.) offset learners they learn x, y distance to the nearest keypoint

c) 3 layer output
   1 layer per distance (distance transform or its inverse or a gaussian)
   1 layer per horizontal offset
   1 layer per vertical offset
   
What should offset learners do if there is no point of interest present?
a) do nothing ?
b) learn to report there is nothing by returning something specific e.g. -1 ?

Learning bbox_centers? (centerness) inner_centers?
"""

b = """ How to represent and learn bounding box, ellipse and minimal rectangle?
Treat it as something real?!
a) 4 layers output (ltrb)
   1 layer for centerness (centerness vs. inner center ?)
   # pleasing aspect is that every pixel predict slightly different value
b) segment
c) 4 layers, each pixel predicts the same 4 numbers
   1 layer for centerness
d) distance transform
e) inverse distance transform
"""

c = """ How to represent and learn scalar region properties?
a) 2 layer output
   1 layer per scalar, the same for all pixels in the designated area
   1 layer per distance transform or its inverse
"""


def get_candidates(
    concepts=[
        "crystal",
        "area_of_interest",
        "loop_inside",
        "loop",
        "stem",
        "pin",
        "ice",
        "capillary",
        "foreground",
        "aether",
        "explorable",
        "support",
        "drop",
        "diffracting_area",  # from raster scans
    ],
    categorical={
        "hierarchy_detailed": [
            "crystal",
            "loop_inside",
            "loop",
            "stem",
            "pin",
            "foreground",
            "background",
        ],
        "hierarchy_crystal_aoi_support_pin": [
            "crystal",
            "area_of_interest",
            "support",
            "pin",
            "foreground",
            "background",
        ],
        "hierarchy_aoi": ["area_of_interest", "foreground", "background"],
        "hierarchy_crystal": ["crystal", "foreground", "background"],
    },
    encoders=["identity", "identity_bw"],
    keypoints=[
        "most_likely_click",
        "extreme",
        "end_likely",
        "start_likely",
        "start_possible",
        "origin",
    ],
    classifications={
        "anything": ["foreground"],
        "plate_content": ["crystals", "precipitate", "other", "clear"],
        "ice": ["ice"],
        "loop_type": ["standard", "mitegen", "crystal_direct", "void"],
    },
    retionprops=["area", "perimeter", "aspect", "extent", "solidity"],
    parameters={
        "binary_segmentation": {"channels": 1, "dtype": "int8"},
        "distance_transform": {"channels": 1, "dtype": "float32"},
        "inverse_distance_transform": {"channels": 1, "dtype": "float32"},
        "sqrt_distance_transform": {"channels": 1, "dtype": "float32"},
        "sqrt_inverse_distance_transform": {"channels": 1, "dtype": "float32"},
        "centerness": {"channels": 1, "dtype": "float32"},
        # ltrb + centerness
        "bounding_box": {"channels": 1 + 4, "dtype": "float32"},
        "min_rectangle": {"channels": 5, "dtype": "float32"},
        "ellipse": {"channels": 5, "dtype": "float32"},
        "moments": {"channels": 9, "dtype": "float32"},
        "regionprops": {"channels": None, "dtype": "float32"},
        "inner_center": {"channels": 1 + 2, "dtype": "float32"},
        "centroid": {"channels": 1 + 2, "dtype": "float32"},
        "bbox_center": {"channels": 1 + 2, "dtype": "float32"},
        "extreme_points": {"channels": 1 + 8, "dtype": "float32"},
        "eigen_points": {"channels": 1 + 8, "dtype": "float32"},
        "encoder": {"channels": None, "dtype": "float32"},
        "categorical_segmentation": {"channels": None, "dtype": "float32"},
        # may make it a variable + centerness ?
        "encoded_shape": {"channels": 1 + 20, "dtype": "float32"},
        "keypoint": {"channels": 1 + 2, "dtype": "float32"},
        "classification": {"channels": None, "dtype": "int8"},
    },
):

    binary_segmentation_concepts = copy.copy(concepts)

    distance_transform_concepts = copy.copy(concepts)
    for concept in ["ice", "diffracting_area"]:
        del distance_transform_concepts[distance_transform_concepts.index(concept)]

    bounding_box_concepts = copy.copy(concepts)
    for concept in ["ice", "diffracting_area", "aether"]:
        del bounding_box_concepts[bounding_box_concepts.index(concept)]

    encoded_shape_concepts = copy.copy(concepts)
    for concept in [
        "ice",
        "diffracting_area",
        "aether",
        "support",
        "explorable",
        "foreground",
    ]:
        del encoded_shape_concepts[encoded_shape_concepts.index(concept)]

    classes_of_tasks = {
        # segmentation
        "binary_segmentation": binary_segmentation_concepts,
        "distance_transform": distance_transform_concepts,
        "inverse_distance_transform": distance_transform_concepts,
        "sqrt_distance_transform": distance_transform_concepts,
        "sqrt_inverse_distance_transform": distance_transform_concepts,
        # regressions
        "bounding_box": bounding_box_concepts,
        # 4 layers output (ltrb),
        # learn associated centerness on the same branch,
        # centerness vs. inner center ?
        "bounding_box_boring": bouding_box_concepts,
        # every pixel within designated area predicts the same 4 numbers
        # 4 layers w, h, x, y
        # 1 layer centerness
        "bounding_box_segment": bounding_box_concepts,
        # learn bounding_box mask in 1 layer
        "bouding_box_distance_transform": bouding_box_concepts,
        # 1 layer
        "bouding_box_inverse_distance_transform": bouding_box_concepts,
        # 1 layer
        "min_rectangle": bounding_box_concepts,
        # 4 layers ltrb within coordinate system of the rectangle
        # 1 layer for orientation
        "min_rectangle_segment": bouding_box_concepts,
        # 1 layer for min_rectangle mask
        "ellipse": bounding_box_concepts,
        # 4 layers ltrb within coordinate system of the ellipse
        # 1 layer for orientation
        "moments": bounding_box_concepts,
        # 9 layers, 1 layer per each moment
        # single number per each pixel within the area
        # 1 layer per distance transform
        "regionprops": bounding_box_concepts,
        # N layers, 1 layer per property,
        # single number per each pixel within the area
        # 1 layer per distance transform
        "inner_center": bounding_box_concepts,
        # modified centerness, offsets, heatmap, distance, (1 - distance), sqrt(1-distance),
        # for point p
        # xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
        # d = np.sqrt((p[0]-yv)**2 + (p[1]-xv)**2)
        # d = d / d.max()
        # d = (1 - d)**2
        # if p not present d = -1
        "centerness": bounding_box_concepts,
        # binary cross entropy, or focal loss
        # modified centerness d = (1 - centerness**2)**2
        "extreme_points": bounding_box_concepts,
        # heatmap for every class of objects and every type of point
        # + offset to the center_of_mass (x, y, 2 layers)
        # + size of the object (width and height, 2 layers)
        # + area of the object (1 layer)
        # inverse distance transform
        # point distance map
        # 2 layers of offsets for each of the point categories
        "eigen_points": bounding_box_concepts,
        # heatmap for every class of objects and every type of point
        # + major_axis, minor_axis (2 layers)
        # + offset to the center of ellipse (x, y, 2 layers)
        # + orientation (8 layers) according to Mousavian, or a single number?
        # + area of the object (1 layer)
        # + euler number (1 layer)
        # + solidity (1 layer)
        # inverse distance transform
        # point distance map
        # 2 layers of offsets for each of the point categories
        "encoded_shape": encoded_shape_concepts,
        # C (e.g. C=21) layer output
        # each pixel within designated area predicts the same value
        # each pixel predicts its distance transform
        # each pixel predicts its inverse distance transform
    }

    candidates = []

    for cot in classes_of_tasks:
        for concept in classes_of_tasks[cot]:
            task = {
                "name": concept,
                "task": cot,
                "dtype": parameters[cot]["dtype"],
                "channels": parameters[cot]["channels"],
                "activation": "sigmoid",
            }
            candidates.append(task)

    cot = "categorical_segmentation"
    for hierarchy, concepts in categorical.items():
        task = {
            "name": hierarchy,
            "task": cot,
            "dtype": parameters[cot]["dtype"],
            "channels": len(concepts),
            "concepts": concepts,
            "activation": "softmax",
        }
        candidates.append(task)

    cot = "encoder"
    for concept in encoders:
        task = {
            "name": concept,
            "task": cot,
            "dtype": parameters[cot]["dtype"],
            "channels": 1 if "bw" in concept else 3,
            "activation": "sigmoid",
        }
        candidates.append(task)

    cot = "keypoint"
    for concept in keypoints:
        task = {
            "name": concept,
            "task": cot,
            "dtype": parameters[cot]["dtype"],
            "channels": parameters[cot]["channels"],
            "activation": "sigmoid",
        }
        candidates.append(task)
        # heatmap for every type of point

    cot = "classification"
    for classification, concepts in classifications.items():
        task = {
            "name": classification,
            "task": cot,
            "dtype": parameters[cot]["dtype"],
            "channels": len(concepts),
            "concepts": concepts,
            "activation": "softmax",
        }
        candidates.append(task)

    return candidates
