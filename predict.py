#!/usr/bin/env python
# -*- coding: utf-8 -*-

import os
import traceback
import zmq
import time
import pickle
import numpy as np
import scipy.ndimage as ndi

from skimage.morphology import remove_small_objects
from skimage.measure import regionprops

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
                   
def get_notion_prediction(predictions, notion, k=0, notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'foreground': 5}, threshold=0.5, min_size=32):
    
    present, r, c, h, w, r_max, c_max, area, notion_prediction = [np.nan] * 9
    
    if type(notion) is list:
        notion_prediction = np.zeros(predictions[0].shape[1:3], dtype=bool)
        for n in notion:
            index = notion_indices[n]
            noti_pred = predictions[index][k,:,:,0]>threshold
            noti_pred = remove_small_objects(noti_pred, min_size=min_size)
            notion_prediction = np.logical_or(notion_prediction, noti_pred)
            
    elif type(notion) is str:
        index = notion_indices[notion]
        
        notion_prediction = predictions[index][k,:,:,0]>threshold
        notion_prediction = remove_small_objects(notion_prediction, min_size=min_size)
        
    if np.any(notion_prediction):
        labeled_image = notion_prediction.astype('uint8')
        properties = regionprops(labeled_image)[0]
        
        if properties.convex_area > min_size:
            present = 1
            area = properties.convex_area
        else:
            present = 0
        bbox = properties.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        r, c = properties.centroid
        c_max = bbox[3]
        r_max = ndi.center_of_mass(labeled_image[:, c_max-5:c_max])[0]
        if notion == 'foreground' or type(notion) is list:
            notion_prediction[bbox[0]: bbox[2], bbox[1]: bbox[3]] = properties.filled_image
        else:
            notion_prediction[bbox[0]: bbox[2], bbox[1]: bbox[3]] = properties.convex_image
    return present, r, c, h, w, r_max, c_max, area, notion_prediction

def get_most_likely_click(predictions):
    shape=predictions[0].shape[1: 3]
    for notion in ['crystal', 'loop']:
        notion_prediction = get_notion_prediction(predictions, notion)
        if notion_prediction[0] == 1:
            return notion_prediction[1]/shape[0], notion_prediction[2]/shape[1]
    foreground = get_notion_prediction(predictions, 'foreground')
    if foreground[0] == 1:
        return foreground[5]/shape[0], foreground[6]/shape[1]
    else:
        return -1, -1
    
    
if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser()
    
    parser.add_argument('-t', '--to_predict', type=str, default='image.jpg', help='to_predict')
    parser.add_argument('-m', '--model', type=str, default='model.h5', help='model')
    parser.add_argument('-H', '--prediction_heigth', default=256, type=int, help='prediction_heigth')
    parser.add_argument('-W', '--prediction_width', default=320, type=int, help='prediction_width')
    parser.add_argument('-s', '--save', default=1, type=int, help='save')
    parser.add_argument('-p', '--prefix', type=str, default='test', help='prefix')
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
    predictions = get_predictions(request_arguments)
    
    print('Client got all predictions')
    print('Most likely click', get_most_likely_click(predictions))
    print()
