#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# part of the MURKO project

notion_importance = {
    "user_click": 0.5,
    "crystal": 1,
    "loop_inside": 2,
    "loop": 3,
    "area_of_interest": 4,
    "stem": 5,
    "support": 6,
    "support_filled": 7,
    "explorable": 7.5,
    "pin": 8,
    "capillary": 9,
    "ice": 10,
    "dust": 11,
    "drop": 12,
    "foreground": 13,
    "not_background": 14,
    "aether": 99,
    "background": 100.0,
}

additional_labels = {
    "cd_loop": "loop",
    "cd_stem": "stem",
    "loop_cd": "loop",
    "stem_cd": "stem",
    "loop_mt": "loop",
    "stem_mt": "stem",
    "mt_loop": "loop",
    "mt_stem": "stem",
}

xkcd_colors_that_i_like = [
    "pale purple",
    "coral",
    "moss green",
    "windows blue",
    "amber",
    "greyish",
    "faded green",
    "dusty purple",
    "crimson",
    "custard",
    "orangeish",
    "dusk blue",
    "ugly purple",
    "carmine",
    "faded blue",
    "dark aquamarine",
    "cool grey",
    "royal blue",
    "light blue",
    "bright green",
    "pale green",
    "pale yellow",
    "lilac",
]

colors_for_labels = {
    "background": "dark green",
    "aether": "light blue",
    "degraded_protein": "chartreuse",
    "crystal_cluster": "greenish yellow",
    "air_bubble": "light aqua",
    "precipitate": "pale yellow",
    "crystalline_precipitate": "pale yellow",
    "crystal_shower": "olive drab",
    "feather-like": "greyish blue",
    "urchin": "light olive",
    "not_background": "greyish",
    "pin": "dusk blue",
    "stem": "crimson",
    "loop": "faded green",
    "loop_inside": "moss green",
    "ice": "ice",
    "dust": "cool grey",
    "capillary": "faded blue",
    "crystal": "banana yellow",
    "drop": "orangeish",
    "support": "dusk blue",
    "support_filled": "orangeish",
    "area_of_interest": "dark green",
    "user_click": "banana yellow",
    "extreme": "dark aquamarine",
    "start_likely": "coral",
    "start_possible": "crimson",
    "most_likely_point": "orangeish",
    "extreme_point": "carmine",
    "start_likely_point": "dusk blue",
    "start_possible_point": "cool grey",
}

line_styles = {
    "loop_inside": "dotted",
    "loop": "dotted",
    "area_of_interest": "dashdot",
    "stem": "dotted",
    "support": "dashed",
    "support_filled": "dashdot",
    "ice": "dotted",
    "crystal": "dotted",
    "pin": "dashed",
    "aether": "dotted",
}


# classes_of_tasks = {
## segmentation
# "binary_segment": binary_segmentation_concepts,
# "distance_transform": distance_transform_concepts,
# "inverse_distance_transform": distance_transform_concepts,
# "sqrt_distance_transform": distance_transform_concepts,
# "sqrt_inverse_distance_transform": distance_transform_concepts,
## regressions
# "bounding_box": bounding_box_concepts,
## 4 layers output (ltrb),
## learn associated centerness on the same branch,
## centerness vs. inner center ?
# "bounding_box_boring": bounding_box_concepts,
## every pixel within designated area predicts the same 4 numbers
## 4 layers w, h, x, y
## 1 layer centerness
# "bounding_box_segment": bounding_box_concepts,
## learn bounding_box mask in 1 layer
# "bounding_box_distance_transform": bounding_box_concepts,
## 1 layer
# "bounding_box_inverse_distance_transform": bounding_box_concepts,
## 1 layer
# "min_rectangle": bounding_box_concepts,
## 4 layers ltrb within coordinate system of the rectangle
## 1 layer for orientation
# "min_rectangle_segment": bounding_box_concepts,
## 1 layer for min_rectangle mask
# "ellipse": bounding_box_concepts,
## 4 layers ltrb within coordinate system of the ellipse
## 1 layer for orientation
# "moments": bounding_box_concepts,
## 9 layers, 1 layer per each moment
## single number per each pixel within the area
## 1 layer per distance transform
# "regionprops": bounding_box_concepts,
## N layers, 1 layer per property,
## single number per each pixel within the area
## 1 layer per distance transform
# "inner_center": bounding_box_concepts,
## modified centerness, offsets, heatmap, distance, (1 - distance), sqrt(1-distance),
## for point p
## xv, yv = np.meshgrid(np.arange(img.shape[1]), np.arange(img.shape[0]))
## d = np.sqrt((p[0]-yv)**2 + (p[1]-xv)**2)
## d = d / d.max()
## d = (1 - d)**2
## if p not present d = -1
# "centerness": bounding_box_concepts,
## binary cross entropy, or focal loss
## modified centerness d = (1 - centerness**2)**2
# "extreme_points": bounding_box_concepts,
## heatmap for every class of objects and every type of point
## + offset to the center_of_mass (x, y, 2 layers)
## + size of the object (width and height, 2 layers)
## + area of the object (1 layer)
## inverse distance transform
## point distance map
## 2 layers of offsets for each of the point categories
# "eigen_points": bounding_box_concepts,
## heatmap for every class of objects and every type of point
## + major_axis, minor_axis (2 layers)
## + offset to the center of ellipse (x, y, 2 layers)
## + orientation (8 layers) according to Mousavian, or a single number?
## + area of the object (1 layer)
## + euler number (1 layer)
## + solidity (1 layer)
## inverse distance transform
## point distance map
## 2 layers of offsets for each of the point categories
# "encoded_shape": encoded_shape_concepts,
## C (e.g. C=21) layer output
## each pixel within designated area predicts the same value
## each pixel predicts its distance transform
## each pixel predicts its inverse distance transform
# }


regionprops = (["area", "perimeter", "aspect", "extent", "solidity"],)

encoders = (["identity", "identity_bw"],)

keypoints = [
    "most_likely_click",
    "origin",
    "extreme",
    "start_possible",
    "start_likely",
    "aoi_base",
    "aoi_top",
    "aoi_left",
    "aoi_right",
    "base_left",
    "base_right",
]

keypoint_labels = dict([(item, i + 1) for i, item in enumerate(keypoints)])

keypoints_global_classification = {
    1: ["most_likely_click", "extreme", "start_likely", "start_possible", "origin"],
    2: [
        "aoi_top",
        "aoi_left",
        "most_likely_click",
        "aoi_right",
        "aoi_base",
        "stem_center",
        "start_possible",
        "pin_left",
        "pin_center",
        "pin_right",
        "origin",
    ],
}

classifications = {
    "anything": ["foreground"],
    "plate_content": ["crystals", "precipitate", "other", "clear"],
    "ice": ["ice"],
    "loop_type": ["standard", "mitegen", "crystal_direct", "void"],
}

categorical = {
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
}


class_tasks = {
    "binary_segment": {"channels": 1, "dtype": "int8"},
    "distance_transform": {"channels": 1, "dtype": "float32"},
    "inverse_distance_transform": {"channels": 1, "dtype": "float32"},
    "sqrt_distance_transform": {"channels": 1, "dtype": "float32"},
    "sqrt_inverse_distance_transform": {"channels": 1, "dtype": "float32"},
}
instance_tasks = {
    "inner_center": {"channels": 1 + 2, "dtype": "float32"},
    "centerness": {"channels": 1, "dtype": "float32"},
    "bounding_box": {"channels": 1 + 4, "dtype": "float32"},
    "min_rectangle": {"channels": 1 + 5, "dtype": "float32"},
    "ellipse": {"channels": 1 + 5, "dtype": "float32"},
    "min_circle": {"channels": 1 + 1, "dtype": "float32"},
    "regionprops": {"channels": 1 + len(regionprops), "dtype": "float32"},
    # centerness + ltrb
    "cltrb": {"channels": 1 + 4, "dtype": "float32"},
    "cltrbo": {"channels": 1 + 5, "dtype": "float32"},
    "moments": {"channels": 1 + 7, "dtype": "float32"},
    "object_points": {"channels": 1 + 8, "dtype": "float32"},
    # may make it a variable + centerness ?
    "encoded_shape": {"channels": 1 + 20, "dtype": "float32"},
}
global_tasks = {
    # voronoi diagram around crystal inner centers
    "crystal_ordinal": {"channels": 100, "dtype": "int8", "activation": "softmax"},
    # centerness + offsets
    "keypoint": {"channels": 1 + 2, "dtype": "float32"},
    # all keypoints together
    "keypoints_regression": {
        "channels": 1 + 2,
        "dtype": "float32",
        "activation": "sigmoid",
    },
    # voronoi diagram
    "keypoints_classification": {
        "channels": len(keypoints),
        "dtype": "int8",
        "activation": "softmax",
    },
    "classification": {"channels": None, "dtype": "int8", "activation": "softmax"},
    "encoder": {"channels": None, "dtype": "float32"},
    "hierarchy": {"channels": None, "dtype": "float32"},
    "moments": {"channels": 7, "dtype": "float32"},
}

concepts = [
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
]
