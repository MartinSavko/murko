#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import os
import zmq
import time
import pickle

from murko import ( 
    get_predictions,
    get_most_likely_click,
    get_loop_bbox
    )

if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--to_predict', type=str, default='image.jpg', help='to_predict')
    parser.add_argument('-H', '--prediction_heigth', default=256, type=int, help='prediction_heigth')
    parser.add_argument('-W', '--prediction_width', default=320, type=int, help='prediction_width')
    parser.add_argument('-s', '--save', action='store_true', help='save')
    parser.add_argument('-P', '--prefix', type=str, default='test', help='prefix')
    parser.add_argument('-p', '--port', type=int, default=8099, help='port')
    args = parser.parse_args()
    
    model_img_size = (args.prediction_heigth, args.prediction_width)
    
    request_arguments = {}
    to_predict = None
    if os.path.isdir(args.to_predict) or os.path.isfile(args.to_predict):
        to_predict = args.to_predict
    
    request_arguments['to_predict'] = to_predict
    request_arguments['model_img_size'] = model_img_size
    request_arguments['save'] = bool(args.save)
    request_arguments['prefix'] = args.prefix
    _start = time.time()
    predictions = get_predictions(request_arguments, port=args.port)
    print('Client got all predictions in %.4f seconds' % (time.time() - _start))
    loop_present, r, c, h, w = get_loop_bbox(predictions)
    if loop_present == 1:
        print('Loop found! Its bounding box parameters in fractional coordianates are: center (vertical %.3f, horizontal %.3f), height %.3f, width %.3f' % (r, c, h, w))
    else:
        print('loop not found.')
    print('Most likely click in fractional coordinates: (vertical %.3f, horizontal %.3f)' % (get_most_likely_click(predictions)))
    
    print()
