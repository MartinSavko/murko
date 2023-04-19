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
import tensorflow as tf

from murko import predict_multihead, get_uncompiled_tiramisu

def get_model(model_name='model.h5'):
    _start_load = time.time()
    for gpu in tf.config.list_physical_devices('GPU'): 
        print('setting memory_growth on', gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
    model = get_uncompiled_tiramisu()
    model.load_weights(model_name)
    _end_load = time.time()
    print('model loaded in %.4f seconds' % (_end_load-_start_load))
    _start_warmup = time.time()
    m = h5py.File(model_name, 'r')
    if 'warmup_image' in m:
        warmup_image = m['warmup_image'][()][0]
        all_predictions = predict_multihead(to_predict=[warmup_image.tobytes()], model_img_size=(256, 320), model=model, save=False)
    m.close()
    
    _end_warmup = time.time()
    print('server warmup run took %.4f seconds' % (_end_warmup - _start_warmup))
    return model

def serve(port=8099, model_name='model.h5'):
    _start = time.time()
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        if tf.config.experimental.get_device_details(gpus[0])['compute_capability'][0]>=7:
            tf.keras.mixed_precision.set_global_policy("mixed_float16")
        for gpu in tf.config.list_physical_devices('GPU'): 
            print('setting memory_growth on', gpu)
            tf.config.experimental.set_memory_growth(gpu, True)
    model = get_model(model_name=model_name)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port )
    print('Model load and warmup took %.4f seconds' % (time.time() - _start))
    print('predict_server ready to serve\n')
    while True:
        request_arguments = socket.recv()
        request = pickle.loads(request_arguments)
        print("%s received request" % (time.asctime(), ))
        try:
            all_predictions = predict_multihead(to_predict=request['to_predict'], model_img_size=request['model_img_size'], model=model, save=request['save'], prefix=request['prefix'])
        except:
            print(traceback.print_exc())
            all_predictions = []
        predictions = pickle.dumps(all_predictions)
        socket.send(predictions)
        print()
        
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-p', '--port', type=int, default=8099, help='port')
    parser.add_argument('-m', '--model_name', type=str, default='model.h5', help='model name')
    parser.add_argument('-d', '--directory', default=None, type=str, help='optional model directory')
    args = parser.parse_args()
    if args.directory is not None:
        model_name = os.path.join(args.directory, args.model_name)
    else:
        model_name = args.model_name
    args = parser.parse_args()
    print('args', args)
    serve(port=args.port, model_name=args.model_name)

