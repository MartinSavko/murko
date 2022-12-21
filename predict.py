#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import zmq
import time
import pickle

from murko import get_most_likely_click

def get_predictions(request_arguments, port=8099, verbose=False):
    start = time.time()
    context = zmq.Context()
    if verbose:
        print('Connecting to server ...')
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://localhost:%d' % port)
    socket.send(pickle.dumps(request_arguments))
    predictions = pickle.loads(socket.recv())
    if verbose:
        print('Received predictions in %.4f seconds' % (time.time() - start))
    return predictions
                   
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--to_predict', type=str, default='image.jpg', help='to_predict')
    parser.add_argument('-m', '--model', type=str, default='model.h5', help='model')
    parser.add_argument('-H', '--prediction_heigth', default=256, type=int, help='prediction_heigth')
    parser.add_argument('-W', '--prediction_width', default=320, type=int, help='prediction_width')
    parser.add_argument('-s', '--save', default=1, type=int, help='save')
    parser.add_argument('-P', '--prefix', type=str, default='test', help='prefix')
    parser.add_argument('-p', '--port', type=int, default=8099, help='port')
    args = parser.parse_args()
    
    print('args', args)
    
    model_img_size = (args.prediction_heigth, args.prediction_width)
    
    request_arguments = {}
    to_predict = None
    if os.path.isdir(args.to_predict) or os.path.isfile(args.to_predict):
        to_predict = args.to_predict
    
    request_arguments['to_predict'] = to_predict
    request_arguments['model_img_size'] = model_img_size
    request_arguments['save'] = bool(args.save)
    request_arguments['prefix'] = args.prefix
    predictions = get_predictions(request_arguments, port=args.port)
    
    print('Client got all predictions')
    print('Most likely click: (vertical %.2f, horizontal %.2f)' % (get_most_likely_click(predictions)))
    print()
