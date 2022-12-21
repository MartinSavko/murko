# Goal
This projects aims to develop a tool to help make a sense of optical images of samples people typically work with in macromolecular crystallography experiments. An approach employed at the current stage is a one using an artificial neural network.

## Requirements
We aim to have a tool that given an image such as this one:
![alt text][input]
[input]: https://github.com/MartinSavko/murko/blob/main/image.jpg "Example image"

will be able to classify all of the pixels as representing salient higher level concepts such as crystal, mother liquor, loop, stem, pin, ice, most likely user click etc ... It should also fulfil the following requirements

* is invariant to scale
* is invariant to wide range of illumination conditions
* is invariant to sample orientation 
** multi axis goniometry
** supporting both horizontally and vertically mounted goniometers
* is fast -- it has to work in real time

## Current performance
this is how it performs at the moment
![alt text][result]
[result]: https://github.com/MartinSavko/murko/blob/main/image_default_model_img_size_256x320_comparison.png "Example result"
