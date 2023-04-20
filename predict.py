#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import os
import zmq
import time
import pickle

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

def get_most_likely_click(predictions, k=0, verbose=False, min_size=32):
    _start = time.time()
    gmlc = False
    most_likely_click = -1, -1
        
    for notion in ['crystal', 'loop_inside', 'loop']:
        notion_prediction = get_notion_prediction(predictions, notion, k=k, min_size=min_size)
        if notion_prediction[0] == 1:
            most_likely_click = notion_prediction[1]/notion_prediction[-1].shape[0], notion_prediction[2]/notion_prediction[-1].shape[1]
            if verbose:
                print('%s found!' % notion)
            gmlc = True
            break
    if gmlc is False:
        foreground = get_notion_prediction(predictions, 'foreground', k=k, min_size=min_size)
        if foreground[0] == 1:
            most_likely_click = foreground[5]/foreground[-1].shape[0], foreground[6]/foreground[-1].shape[1]
    if verbose:
        print('most likely click determined in %.4f seconds' % (time.time() - _start))
    return most_likely_click

def get_loop_bbox(predictions, k=0, min_size=32):
    loop_present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, notion_prediction = get_notion_prediction(predictions, ['crystal', 'loop_inside', 'loop'], k=k, min_size=min_size)
    shape = predictions[0].shape[1:3]
    if bbox is not np.nan:
        r = bbox[0] + h/2
        c = bbox[1] + w/2
    r /= shape[0]
    c /= shape[1]
    h /= shape[0]
    w /= shape[1]
    return loop_present, r, c, h, w

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--to_predict', type=str, default='image.jpg', help='to_predict')
    parser.add_argument('-H', '--prediction_heigth', default=256, type=int, help='prediction_heigth')
    parser.add_argument('-W', '--prediction_width', default=320, type=int, help='prediction_width')
    parser.add_argument('-s', '--save', action='store_true', help='save')
    parser.add_argument('-P', '--prefix', type=str, default='test', help='prefix')
    parser.add_argument('-p', '--port', type=int, default=8099, help='port')
    parser.add_argument('-m', '--min_size', type=int, default=64, help='minimum object size')

    args = parser.parse_args()
    
    model_img_size = (args.prediction_heigth, args.prediction_width)
    
    request_arguments = {}
    to_predict = None
    if os.path.isdir(args.to_predict) or os.path.isfile(args.to_predict):
        to_predict = os.path.realpath(args.to_predict)
    
    request_arguments['to_predict'] = to_predict
    request_arguments['model_img_size'] = model_img_size
    request_arguments['save'] = bool(args.save)
    request_arguments['prefix'] = args.prefix
    
    print('request_arguments: %s' % request_arguments)
    _start = time.time()
    predictions = get_predictions(request_arguments, port=args.port)
    print('Client got all predictions in %.4f seconds' % (time.time() - _start))
    loop_present, r, c, h, w = get_loop_bbox(predictions, min_size=args.min_size)
    if loop_present == 1:
        print('Loop found! Its bounding box parameters in fractional coordianates are: center (vertical %.3f, horizontal %.3f), height %.3f, width %.3f' % (r, c, h, w))
    else:
        print('Loop not found.')
    print('Most likely click in fractional coordinates: (vertical %.3f, horizontal %.3f)' % (get_most_likely_click(predictions, min_size=args.min_size)))
    
    print()
