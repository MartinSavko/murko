#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)

import os
import zmq
import time
import pickle
import numpy as np
from skimage.morphology import remove_small_objects
from skimage.measure import regionprops
import scipy.ndimage as ndi

def get_predictions(request_arguments, port=8099, verbose=False):
    start = time.time()
    context = zmq.Context()
    if verbose:
        print('Connecting to server ...')
    socket = context.socket(zmq.REQ)
    socket.connect('tcp://localhost:%d' % port)
    socket.send(pickle.dumps(request_arguments))
    raw_predictions = socket.recv()
    print('client received raw_predictions')
    print(type(raw_predictions))
    print(raw_predictions[:10])
    predictions = pickle.loads(str(raw_predictions))
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

def get_notion_description(mask, min_size=32):
    present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, properties = [np.nan] * 12
    present = -1
    
    if np.any(mask):
        labeled_image = mask.astype('uint8')
        properties = regionprops(labeled_image)[0]
        
        if properties.convex_area > min_size:
            present = 1
            area = properties.convex_area
        else:
            present = -1
        bbox = properties.bbox
        h = bbox[2] - bbox[0]
        w = bbox[3] - bbox[1]
        r, c = properties.centroid
        c_max = bbox[3]
        r_max = ndi.center_of_mass(labeled_image[:, c_max-5:c_max])[0]
        c_min = bbox[1]
        r_min = ndi.center_of_mass(labeled_image[:, c_min: c_min+5])[0]

    return present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, properties


def get_notion_mask_from_predictions(predictions, notion, k=0, notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'foreground': 5}, threshold=0.5, min_size=32):
        
    notion_mask = np.zeros(predictions[0].shape[1:3], dtype=bool)
    
    if type(notion) is list:
        for n in notion:
            index = notion_indices[n]
            noti_pred = predictions[index][k,:,:,0]>threshold
            noti_pred = remove_small_objects(noti_pred, min_size=min_size)
            notion_mask = np.logical_or(notion_mask, noti_pred)
            
    elif type(notion) is str:
        index = notion_indices[notion]
        notion_mask = predictions[index][k,:,:,0]>threshold
        notion_mask = remove_small_objects(notion_mask, min_size=min_size)
    return notion_mask
    
    
def get_notion_mask_from_masks(masks, notion, notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'foreground': 5}, min_size=32):
    
    notion_mask = np.zeros(masks.shape[:2], dtype=bool)
    
    if type(notion) is list:
        for n in notion:
            index = notion_indices[n]
            noti_mask = masks[:,:,index]
            noti_mask = remove_small_objects(noti_mask>0, min_size=min_size)
            notion_mask = np.logical_or(notion_mask, noti_mask)
    elif type(notion) is str:
        index = notion_indices[notion]
        notion_mask = masks[:,:,index]
        notion_mask = remove_small_objects(notion_mask>0, min_size=min_size)
    return notion_mask

def get_notion_prediction(predictions, notion, k=0, notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'foreground': 5}, threshold=0.5, min_size=32):
    
    if type(predictions) is list:
        notion_mask = get_notion_mask_from_predictions(predictions, notion, k=k, notion_indices=notion_indices, threshold=threshold, min_size=min_size)
    elif type(predictions) is np.ndarray and len(predictions.shape) == 3:
        notion_mask = get_notion_mask_from_masks(predictions, notion, notion_indices=notion_indices, min_size=min_size)
        
    present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, properties = get_notion_description(notion_mask, min_size=min_size)
    
    if type(properties) != float:
        if notion == 'foreground' or type(notion) is list:
            notion_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = properties.filled_image
        else:
            notion_mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = properties.convex_image
    
    return present, r, c, h, w, r_max, c_max, r_min, c_min, bbox, area, notion_mask

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
