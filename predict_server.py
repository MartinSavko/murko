#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import zmq
import time
import sys
import json
import pickle
import traceback
from murko import predict_multihead, get_uncompiled_tiramisu


def get_model(model_name='model.h5'):
    _start = time.time()
    model = get_uncompiled_tiramisu()
    model.load_weights(model_name)
    _end = time.time()
    print('model loaded in %.4f seconds' % (_end-_start))
    return model

def serve(port=8099, model_name='model.h5'):
    model = get_model(model_name=model_name)
    context = zmq.Context()
    socket = context.socket(zmq.REP)
    socket.bind("tcp://*:%s" % port )
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
    parser.add_argument('-m', '--model_name', type=str, default='model.h5', help='model')
    
    args = parser.parse_args()
    
    serve(port=args.port, model_name=args.model_name)

