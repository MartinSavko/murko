#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import os
import zmq
import time
import pickle


def get_predictions(request_arguments, host="localhost", port=8901, verbose=False):
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


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "-t", "--to_predict", type=str, default="image.jpg", help="to_predict"
    )
    parser.add_argument(
        "-H", "--prediction_heigth", default=256, type=int, help="prediction_heigth"
    )
    parser.add_argument(
        "-W", "--prediction_width", default=320, type=int, help="prediction_width"
    )
    parser.add_argument("-s", "--save", action="store_true", help="save")
    parser.add_argument("-P", "--prefix", type=str, default="test", help="prefix")
    parser.add_argument("-p", "--port", type=int, default=8901, help="port")
    parser.add_argument("-o", "--host", type=str, default="localhost", help="host")
    parser.add_argument(
        "-m", "--min_size", type=int, default=64, help="minimum object size"
    )

    args = parser.parse_args()
    print(args)

    model_img_size = (args.prediction_heigth, args.prediction_width)

    request_arguments = {}
    to_predict = None
    if os.path.isdir(args.to_predict) or os.path.isfile(args.to_predict):
        to_predict = os.path.realpath(args.to_predict)

    request_arguments["to_predict"] = to_predict
    request_arguments["model_img_size"] = model_img_size
    request_arguments["save"] = bool(args.save)
    request_arguments["min_size"] = args.min_size
    request_arguments["description"] = [
        "foreground",
        "crystal",
        "loop_inside",
        "loop",
        ["crystal", "loop"],
        ["crystal", "loop", "stem"],
    ]
    request_arguments["prefix"] = args.prefix

    print("request_arguments: %s" % request_arguments)
    _start = time.time()
    analysis = get_predictions(request_arguments, host=args.host, port=args.port)

    print("Client got all predictions in %.4f seconds" % (time.time() - _start))
    description = analysis["descriptions"][0]

    if description["aoi_bbox"][0] == 1:
        r, c, h, w = description["aoi_bbox"][1:]
        print(
            "Loop found! Its bounding box parameters in fractional coordianates are: center (vertical %.3f, horizontal %.3f), height %.3f, width %.3f"
            % (r, c, h, w)
        )
    elif description["foreground"]["present"] != 0:
        print("Loop not detected but sample seems to be present!")
    else:
        print("No sample detected ...")

    print(
        "Most likely click in fractional coordinates: vertical %.3f, horizontal %.3f"
        % (description["most_likely_click"][0], description["most_likely_click"][1])
    )

    print()
