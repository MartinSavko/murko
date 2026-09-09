#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import os
import h5py
import zmq
import time
import sys
import json
import pickle
import traceback
import numpy as np
import tensorflow as tf
from tensorflow import keras
import psutil
import gc

import simplejpeg
from imageio import imread

from murko import (
    WSConv2D,
    WSSeparableConv2D,
)

from sample import get_resized_image
from utils import get_descriptions, plot_analysis
from config import luts

def print_memory_use():
    # https://stackoverflow.com/questions/44327803/memory-leak-with-tensorflow
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0] / 2.0**30  # memory use in GB...I think
    print("memory use: %.3f GB" % memoryUse)


def get_model(model_name="model.h5", model_img_size=(128, 128), gpu="0", integrate=True):
    _start_load = time.time()

    if "CUDA_VISIBLE_DEVICES" not in os.environ:
        os.environ["CUDA_VISIBLE_DEVICES"] = gpu
    gpus = tf.config.list_physical_devices("GPU")
    print("gpu found", gpus)
    if gpus:
        if (
            tf.config.experimental.get_device_details(gpus[0])["compute_capability"][0]
            >= 7
        ):
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        for gpu in tf.config.list_physical_devices("GPU"):
            print("setting memory_growth on", gpu)
            tf.config.experimental.set_memory_growth(gpu, True)

    model = keras.models.load_model(
        model_name,
        custom_objects={
            "WSConv2D": WSConv2D,
            "WSSeparableConv2D": WSSeparableConv2D,
        },
    )

    output_names = model.output_names
    if integrate:
        inputs = keras.layers.Input((None, None, 3))
        resized = keras.layers.Resizing(model_img_size[0], model_img_size[1], interpolation='bilinear')(inputs)
        rescaled = keras.layers.Rescaling(scale=1.0 / 255)(resized)
        outputs = model(rescaled)
        model = keras.Model(inputs=inputs, outputs=outputs)
        model.output_names = output_names

    _end_load = time.time()
    print("model loaded in %.3f seconds" % (_end_load - _start_load))
    _start_warmup = time.time()

    to_predict = np.zeros((1,) + model_img_size + (3,), dtype="float32")

    predictions = model.predict(to_predict, batch_size=1)
    _end_warmup = time.time()

    del predictions

    gc.collect()
    print_memory_use()
    print("server warmup run took %.3f seconds" % (_end_warmup - _start_warmup))
    return model


def serve(
    port=8901,
    model_name="model.keras",
    gpu="0",
    batch_size=1,
    model_img_size=(128, 128),
    min_size = 32,
    debug=True,
    default_hierarchy_output_name="hierarchy_detailed_hierarchy",
    integrate=True,
):
    _start = time.time()

    model = get_model(
        model_name=model_name,
        gpu=gpu,
        model_img_size=model_img_size,
        integrate=integrate,
    )

    notion_indices = dict([(item, k) for k, item in enumerate(model.output_names)])

    print(5*"\n")
    print(f"notion_indices\n{notion_indices}")
    print(5*"\n")

    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port)

    print("Model load and warmup took %.3f seconds" % (time.time() - _start))
    print("murko ready to serve\n")
    while True:
        requests = socket.recv()
        request = pickle.loads(requests)
        _start = time.time()
        print("%s received request" % (time.asctime(),))
        analysis = {}
        try:
            to_predict = request["to_predict"]
            image_paths = []
            if "min_size" in request:
                min_size = request["min_size"]

            if debug:
                print("debug type(to_predict)", type(to_predict))

            if isinstance(to_predict, bytes) and simplejpeg.is_jpeg(to_predict):
                to_predict = [simplejpeg.decode_jpeg(to_predict)]

            elif isinstance(to_predict, str) and (
                to_predict.lower().endswith(".jpg")
                or to_predict.lower().endswith(".jpeg")
            ):
                image_paths = [to_predict[:]]
                to_predict = [simplejpeg.decode_jpeg(open(to_predict, "rb").read())]

            elif isinstance(to_predict, str) and to_predict.lower().endswith(".png"):
                image_paths = [to_predict[:]]
                to_predict = [imread(to_predict)]

            elif isinstance(to_predict, list) and os.path.isfile(to_predict[0]):
                image_paths = to_predict[:]
                to_predict = [
                    simplejpeg.decode_jpeg(open(item, "rb").read())
                    for item in to_predict
                ]

            elif isinstance(to_predict, list) and len(to_predict[0].shape) != 3:
                to_predict = [
                    simplejpeg.decode_jpeg(jpeg) for jpeg in to_predict
                ]

            original_image_shape = to_predict[0].shape
            analysis["original_image_shape"] = original_image_shape
            if debug:
                print("to_predict type before prep", type(to_predict[0]))

            if not integrate:
                to_predict_unresized = to_predict.copy()
                _start_prep = time.time()
                to_predict = [
                    (get_resized_image(item, model_img_size) / 255.0).astype("float32")
                    for item in to_predict
                ]

                _end_prep = time.time()
                print(f"images rescaled and resized in {_end_prep - _start_prep:.3f} seconds")
            else:
                to_predict_unresized = to_predict

            to_predict = np.array(to_predict)

            if len(to_predict.shape) == 3:
                to_predict = np.expand_dims(to_predict, 0)

            if debug:
                print("to_predict type after prep", type(to_predict[0]))
                print("to_predict.shape", to_predict.shape)

            all_predictions = model.predict(
                to_predict, batch_size=min([len(to_predict), batch_size])
            )

            duration = time.time() - _start
            N = len(all_predictions[0])
            print(
                "%d predictions took %.3f seconds (%.3f per image)"
                % (N, duration, duration / N)
            )
            descriptions = []
            if "hierarchy_output_name" in request:
                hierarchy_output_name = request["hierarchy_output_name"]
            else:
                hierarchy_output_name = default_hierarchy_output_name

            lut_hierarchy_key = hierarchy_output_name.replace("_hierarchy", "")
            lut = luts[lut_hierarchy_key]

            analysis["predictions"] = all_predictions
            if "description" in request and request["description"] is not False:
                _start_description = time.time()
                try:
                    descriptions = get_descriptions(
                        all_predictions,
                        notions=request["description"],
                        notion_indices=notion_indices,
                        original_image_shape=original_image_shape,
                        min_size=min_size,
                        hierarchy_output_name=hierarchy_output_name,
                        lut=lut,
                    )
                    analysis["descriptions"] = descriptions
                    print(
                        "descriptions took %.3f seconds"
                        % (time.time() - _start_description)
                    )
                except:
                    traceback.print_exc()
                    print("problem in getting descriptions, please check !")
                    analysis = all_predictions

                if "save" in request and request["save"]:
                    plot_analysis(to_predict_unresized, analysis, image_paths=image_paths)

            del all_predictions
            if descriptions:
                del descriptions

        except:
            traceback.print_exc()

        socket.send(pickle.dumps(analysis))

        print("complete analysis took %.3f seconds" % (time.time() - _start))
        del analysis
        gc.collect()
        print_memory_use()
        print()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument("-p", "--port", type=int, default=8901, help="port")
    parser.add_argument(
        "-m", "--model_name", type=str, default="model.keras", help="model"
    )
    parser.add_argument(
        "-s",
        "--model_img_size",
        type=str,
        default="(128, 128)",
        help="working image resolution",
    )
    parser.add_argument(
        "-d", "--directory", default=None, type=str, help="optional model directory"
    )
    parser.add_argument("-g", "--gpu", default="0", type=str, help="gpu to use")

    parser.add_argument(
        "-i", "--integrate", action="store_true", help="integrate resize and scale layers")

    args = parser.parse_args()
    model_img_size = eval(args.model_img_size)
    if not os.path.isfile(args.model_name) and args.directory is not None:
        model_name = os.path.join(args.directory, args.model_name)
    else:
        model_name = args.model_name
    args = parser.parse_args()
    print("args", args)
    serve(
        port=args.port,
        model_name=model_name,
        model_img_size=model_img_size,
        gpu=args.gpu,
        integrate=args.integrate,
    )
