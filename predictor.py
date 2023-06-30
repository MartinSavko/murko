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
from murko import get_uncompiled_tiramisu, get_notion_prediction

def get_raw_projections(predictions, notion='foreground', notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'foreground': 5}, threshold=0.5, min_size=32):
    raw_projections = []
    for k in range(len(predictions[notion_indices[notion]])):
        present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, notion_mask = get_notion_prediction(predictions, notion, k=k)
        if present:
            raw_projections.append((present, (r, c, h, w), notion_mask))
    return raw_projections

def print_memory_use():
    # https://stackoverflow.com/questions/44327803/memory-leak-with-tensorflow
    pid = os.getpid()
    py = psutil.Process(pid)
    memoryUse = py.memory_info()[0]/2.**30  # memory use in GB...I think
    print('memory use:', memoryUse)
    
def get_model(model_name='model.h5', model_img_size=(256, 320), default_gpu='0'):
    _start_load = time.time()

    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = default_gpu
    gpus = tf.config.list_physical_devices('GPU')
    for gpu in gpus: 
        print('setting memory_growth on', gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    model = get_uncompiled_tiramisu()
    model.load_weights(model_name)
    
    inputs = keras.layers.Input((None, None, 3))
    resized = keras.layers.Resizing(model_img_size[0], model_img_size[1])(inputs)
    rescaled = keras.layers.Rescaling(scale=1./255)(resized)
    outputs = model(rescaled)
    integrated_resize_model = keras.Model(inputs=inputs, outputs=outputs)
    
    _end_load = time.time()
    print('model loaded in %.4f seconds' % (_end_load-_start_load))
    _start_warmup = time.time()
    m = h5py.File(model_name, 'r')
    if 'warmup_image' in m:
        to_predict = np.expand_dims(simplejpeg.decode_jpeg(m['warmup_image'][()][0].tobytes()),0)
    else:
        to_predict = np.zeros((1,) + model_img_size+(3,), dtype='uint8')
    m.close()
    
    predictions = integrated_resize_model.predict(to_predict)
    _end_warmup = time.time()
    print('server warmup run took %.4f seconds' % (_end_warmup - _start_warmup))
    return integrated_resize_model

def serve(port=8901, model_name='model.h5', default_gpu='0', batch_size=16, model_img_size=(256, 320)):
    _start = time.time()
    if 'CUDA_VISIBLE_DEVICES' not in os.environ:
        os.environ['CUDA_VISIBLE_DEVICES'] = default_gpu
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if tf.config.experimental.get_device_details(gpus[0])['compute_capability'][0]>=7:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        for gpu in tf.config.list_physical_devices('GPU'): 
            print('setting memory_growth on', gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
    model = get_model(model_name=model_name, default_gpu=default_gpu, model_img_size=model_img_size)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port )
    print('Model load and warmup took %.4f seconds' % (time.time() - _start))
    print('predict_server ready to serve\n')
    while True:
        requests = socket.recv()
        request = pickle.loads(requests) 
        _start = time.time()
        print("%s received request" % (time.asctime(), ))
        to_predict = request['to_predict']
        if type(to_predict) is list and len(to_predict[0].shape) != 3 :
            try:
                to_predict = np.array([simplejpeg.decode_jpeg(jpeg) for jpeg in to_predict])
            except:
                pass
        elif type(to_predict) is np.ndarray and len(to_predict.shape) == 3:
            to_predict = np.expand_dims(to_predict, 0)
        analysis = {}
        analysis['original_image_shape'] = to_predict[0].shape
        raw_projections={}
        try:
            all_predictions = model.predict(to_predict, batch_size=min([len(to_predict), batch_size]))
        except:
            print(traceback.print_exc())
            all_predictions = []
        analysis['predicitons'] = all_predictions
        if 'raw_projections' in request and request['raw_projections'] is not False:
            for notion in request['raw_projections']:
                raw_projections[','.join(notion) if type(notion) is list else notion] = get_raw_projections(all_predictions, notion=notion)
            analysis['raw_projections'] = raw_projections
        
        socket.send(pickle.dumps(analysis))
        print('all predictions took %.4f seconds' % (time.time() - _start))
        del all_predictions
        del raw_projections
        del analysis
        gc.collect()
        print_memory_use()
        print()
        
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p', '--port', type=int, default=8901, help='port')
    parser.add_argument('-m', '--model_name', type=str, default='model.h5', help='model')
    parser.add_argument('-s', '--model_img_size', type=str, default='(256, 320)', help='working image resolution')
    parser.add_argument('-d', '--directory', default=None, type=str, help='optional model directory')
    args = parser.parse_args()
    model_img_size = eval(args.model_img_size)
    if not os.path.isfile(args.model_name) and args.directory is not None:
        model_name = os.path.join(args.directory, args.model_name)
    else:
        model_name = args.model_name
    args = parser.parse_args()
    print('args', args)
    serve(port=args.port, model_name=model_name, model_img_size=model_img_size)

