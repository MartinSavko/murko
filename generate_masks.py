#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import os
import glob
import time
from convert_labelme_json_to_masks import convert, generate_background_masks


def main(
    source="soleil_proxima_dataset",
    destination="images_and_labels",
    check_file="img.jpg",
):
    _start = time.time()
    if not os.path.isdir(destination):
        os.makedirs(destination)
    labels = glob.glob(os.path.join(source, "*.json"))
    for json_file in labels:
        dest = os.path.join(destination, os.path.basename(json_file)[:-5])
        if not os.path.isfile(os.path.join(dest, check_file)):
            convert(json_file, out=dest)
    _end = time.time()
    duration = _end - _start
    print(
        "Converting %d labels took %.4f seconds (%.4f seconds per file)"
        % (len(labels), duration, duration / len(labels))
    )

    _start = time.time()
    print("generating background masks")
    backgrounds = glob.glob(os.path.join(source, "Backgrounds", "*.jpg"))
    for background in backgrounds:
        generate_background_masks(background, destination=destination)
    _end = time.time()
    print(
        "Generating %d background masks took %.4f seconds (%.4f seconds per file)"
        % (len(backgrounds), duration, duration / len(backgrounds))
    )


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--source", type=str, default="soleil_proxima_dataset")
    parser.add_argument("-d", "--destination", type=str, default="images_and_labels")
    parser.add_argument("-c", "--check_file", type=str, default="img.jpg")
    args = parser.parse_args()

    print(args)
    main(source=args.source, destination=args.destination, check_file=args.check_file)
