#!/usr/bin/env python
# -*- coding: utf-8 -*-

import copy
from config import (
    regionprops,
    encoders,
    keypoints,
    keypoint_labels,
    classifications,
    concepts,
    categorical,
    class_tasks,
    instance_tasks,
    global_tasks,
    keypoints_global_classification,
)

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


def get_candidates(concepts=concepts):

    task_concepts = {}

    # class_tasks
    task_concepts["binary_segment"] = copy.copy(concepts)

    distance_transform_concepts = copy.copy(concepts)
    for concept in ["ice", "diffracting_area"]:
        del distance_transform_concepts[distance_transform_concepts.index(concept)]
    for dt in [
        "distance_transform",
        "inverse_distance_transform",
        "sqrt_distance_transform",
        "sqrt_inverse_distance_transform",
    ]:
        task_concepts[dt] = distance_transform_concepts

    # instance_tasks
    bounding_box_concepts = copy.copy(concepts)
    for concept in ["ice", "diffracting_area", "aether"]:
        del bounding_box_concepts[bounding_box_concepts.index(concept)]
    for it in instance_tasks:
        if it != "encoded_shape":
            task_concepts[it] = bounding_box_concepts

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
    task_concepts["encoded_shape"] = encoded_shape_concepts

    # global_tasks nothing to do

    # all tasks
    tasks = {}
    for t in (class_tasks, instance_tasks, global_tasks):
        tasks.update(t)

    candidates = []

    for task in tasks:
        if task in task_concepts:
            for concept in task_concepts[task]:
                head = {
                    "name": f"{concept}_{task}",
                    "task": task,
                    "dtype": tasks[task]["dtype"],
                    "channels": tasks[task]["channels"],
                    "activation": "sigmoid",
                }
                candidates.append(head)
        elif task == "hierarchy":
            for hierarchy, concepts in categorical.items():
                head = {
                    "name": hierarchy,
                    "task": task,
                    "dtype": tasks[task]["dtype"],
                    "channels": len(concepts),
                    "concepts": concepts,
                    "activation": "softmax",
                }
                candidates.append(head)

        elif task == "encoder":
            for concept in encoders:
                head = {
                    "name": concept,
                    "task": task,
                    "dtype": tasks[task]["dtype"],
                    "channels": 1 if "bw" in concept else 3,
                    "activation": "sigmoid",
                }
                candidates.append(head)

        elif task == "keypoint":
            print(task)
            for concept in keypoints:
                # heatmap and offsets for every type of point
                print(f"concept {concept}")
                head = {
                    "name": concept,
                    "task": task,
                    "dtype": tasks[task]["dtype"],
                    "channels": tasks[task]["channels"],
                    "activation": "sigmoid",
                }
                candidates.append(head)

        elif task in ["keypoints_regression", "keypoints_classification"]:
            for k in keypoints_global_classification:
                head = {
                    "name": f"task_{k}",
                    "task": task,
                    "dtype": tasks[task]["dtype"],
                    "channels": len(keypoints),
                    "concepts": keypoints,
                    "activation": tasks[task]["activation"],
                }
                candidates.append(head)

        elif task == "classification":
            for classification, concepts in classifications.items():
                head = {
                    "name": classification,
                    "task": task,
                    "dtype": tasks[task]["dtype"],
                    "channels": len(concepts),
                    "concepts": concepts,
                    "activation": "softmax",
                }
                candidates.append(head)

        elif task == "crystal_ordinal":
            head = {
                "name": task,
                "task": "classification",
                "dtype": tasks[task]["dtype"],
                "channels": tasks[task]["channels"],
                "activation": "softmax",
            }
            candidates.append(head)

    return candidates, task_concepts


if __name__ == "__main__":
    c = get_candidates()
    print(c)
