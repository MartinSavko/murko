# Goal
This projects aims to develop a tool to help make sense of optical images of samples people typically work with in macromolecular crystallography experiments. An approach employed at the current stage is the one using an artificial neural network. The current model is based on the architecture, normalization technique, loss definition and other key ideas from the research described in the following papers: 
* The One Hundred Layers Tiramisu: Fully convolutional DenseNets for Semantic Segmentation, [arXiv:1611.09326](https://arxiv.org/abs/1611.09326])
* Xception: Deep Learning with Depthwise Separable Convolutions, [arXiv:1610.02357](https://arxiv.org/abs/1610.02357)
* Micro-Batch Training with Batch-Channel Normalization and Weight Standardization [arXiv:1903.10520](https://arxiv.org/abs/)
* Focal loss: Focal loss for Dense Object Detection [arXiv:1708.02002](https://arxiv.org/abs/1708.02002)

## Requirements
We aim to have a tool that given an image such as this one:
![Example input](https://github.com/MartinSavko/murko/blob/main/examples/image.jpg)

will be able to classify all of the pixels as representing salient higher level concepts such as crystal, mother liquor, loop, stem, pin, ice, most likely user click etc ... It should also fulfil the following requirements

* is invariant to scale
* is invariant to wide range of illumination conditions
* is invariant to sample orientation 
  * supporting multi axis goniometry
  * supporting both horizontally and vertically mounted goniometers
* is fast -- it has to work in real time

## Current performance
This is how it performs at the moment
![Result](https://github.com/MartinSavko/murko/blob/main/examples/image_default_model_img_size_256x320_comparison.png)

More details can be gleaned from the following [presentation](https://bit.ly/murko_isac).
If you find the code useful or want to learn more about how to deploy it at your beamline please drop me a line.

## Usage
1. Start server
```
./predict_server.py
```
model loading and warmup run will take about 10 seconds.

2. query the server
```
./predict.py -t examples/image.jpg --save

```
In practice you will most likely use from your own python client. You might have a look in predict.py to get more precise idea of how to use it. Here is an example
```python
from murko import ( 
    get_predictions,
    get_most_likely_click,
    get_loop_bbox
    )
request_arguments = {}
request_arguments['to_predict'] = 'examples/image.jpg' # may be a path to an image, directory, jpeg string, list of jpegs, list of ndarrays, to_predict, etc... (have a look at segment_multihead() method in murko.py to see how is it handled
request_arguments['model_img_size'] = model_img_size # what resolution will be the prediction run at. May be arbitrary, (256, 320) is the default.
request_arguments['save'] = True # Whether to save predictions or not.
port = 8099 # port on which the server is listening
predicitions = get_predictions(request_arguments, port=port)

most_likely_click = get_most_likely_click(predictions)
loop_present, r, c, h, w = get_loop_bbox(predictions)
```
