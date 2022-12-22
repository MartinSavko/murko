#!/usr/bin/env python
# -*- coding: utf-8 -*-
# author: Martin Savko (martin.savko@synchrotron-soleil.fr)
# based on F. Chollet's https://keras.io/examples/vision/oxford_pets_image_segmentation/
# Model based on The One Hundred Layers Tiramisu: Fully convolutional DenseNets for Semantic Segmentation, arXiv:1611.09326
# With main difference being use of SeparableConv2D instead of Conv2D and using GroupNormalization instead of BatchNormalization. Plus using additional Weight standardization (based on Qiao et al. Micro-Batch Training with Batch-Channel Normalization and Weight Standardization arXiv:1903.10520v2)

import sys
import os
import glob
import numpy as np
import random
import re
import pickle
import traceback
import pylab
import seaborn as sns
import simplejpeg
import scipy.ndimage as ndi
sns.set(color_codes=True)
#from matplotlib import rc
#rc('font', **{'family':'serif','serif':['Palatino']})
#rc('text', usetex=True)

import math

import tensorflow as tf
import tensorflow_addons as tfa
import tensorflow.experimental.numpy as tnp
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import regularizers 
from tensorflow.keras.preprocessing.image import save_img, load_img, img_to_array, array_to_img
from tensorflow.keras.preprocessing import image

from keras.models import Model, load_model
from keras.layers import Input
from keras.layers.core import Dropout, Lambda
from keras.layers.convolutional import Conv2D, SeparableConv2D, Conv2DTranspose
from keras.layers.pooling import MaxPooling2D
#from keras.layers.merge import concatenate
from keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
from keras import backend as K

from skimage.io import imread, imsave
from skimage.transform import resize
from skimage.measure import regionprops
from skimage.morphology import remove_small_objects
try:
    from skimage.morphology.footprints import disk
except:
    from skimage.morphology.selem import disk
import matplotlib.pyplot as plt
import time

from keras.metrics import BinaryCrossentropy
from keras.losses import LossFunctionWrapper
from keras.utils import losses_utils
from tensorflow.python.util import dispatch

directory = 'images_and_labels_augmented'
img_size = (1024, 1360)
model_img_size = (512, 512)
num_classes = 1
batch_size = 8
params = {'segmentation': {'loss': 'binary_focal_crossentropy', 'metrics': 'BIoU'},
          'click_segmentation': {'loss': 'binary_focal_crossentropy', 'metrics': 'BIoUm'},
          'click_regression': {'loss': 'mean_squared_error', 'metrics': 'mean_absolute_error'}}

networks = {
            'fcdn103': {'growth_rate': 16, 'layers_scheme': [4, 5, 7, 10, 12], 'bottleneck': 15},
            'fcdn67':  {'growth_rate': 16, 'layers_scheme': [5]*5, 'bottleneck': 5},
            'fcdn56':  {'growth_rate': 12, 'layers_scheme': [4]*5, 'bottleneck': 4}
        }

calibrations = \
           {1: np.array([0.00160829, 0.001612  ]),
            2: np.array([0.00129349, 0.0012945 ]),
            3: np.array([0.00098891, 0.00098577]),
            4: np.array([0.00075432, 0.00075136]),
            5: np.array([0.00057437, 0.00057291]),
            6: np.array([0.00043897, 0.00043801]),
            7: np.array([0.00033421, 0.00033406]),
            8: np.array([0.00025234, 0.00025507]),
            9: np.array([0.00019332, 0.00019494]),
            10: np.array([0.00015812, 0.00015698])}

'''
dataset composition:
      notion         fraction_label      fraction_total
 
     crystal:           0.051084,           0.023338
 loop_inside:           0.118100,           0.053953
        loop:           0.294535,           0.134557
        stem:           0.062713,           0.028650
         pin:           0.035046,           0.016011
   capillary:           0.004127,           0.001885
         ice:           0.026989,           0.012330
  foreground:           0.407406,           0.186121
  background:           1.781519,           0.813879


'''
#loss_weights_from_stats =\
    #{'crystal': 8,
    #'loop_inside': 3.5,
    #'loop': 1.5,
    #'stem': 6.5,
    #'pin': 5.0,
    #'capillary': 1.,
    #'ice': 1.,
    #'foreground': 1.0,
    #'click': 1.}
    
'''
total pixels 1805946880 (1.806G)
total foreground 319001744 (0.319G, 0.1766 of all)

         notion  fraction_label  fraction_total         weight_label         weight_total
        crystal       0.1173            0.0207                8.5                48.3
    loop_inside       0.2485            0.0439                4.0                22.8
           loop       0.6427            0.1135                1.6                 8.8
           stem       0.1876            0.0331                5.3                30.2
            pin       0.1242            0.0219                8.0                45.6
      capillary       0.0102            0.0018               98.4               557.0
            ice       0.0630            0.0111               15.9                89.9
     foreground       1.0000            0.1766                1.0                 5.7
          total       5.6612            1.0000                0.2                 1.0
'''

loss_weights_from_stats =\
    {'crystal': 8.5,
     'loop_inside': 4.0,
     'loop': 1.6,
     'stem': 5.3,
     'pin': 8.0,
     'capillary': 1.,
     'ice': 15.9,
     'foreground': 1.0,
     'click': 1.0}

from tensorflow.python.util.tf_export import keras_export
@keras_export('keras.losses.BinaryFocalCrossentropy')
class BinaryFocalCrossentropy(LossFunctionWrapper):
  """Computes the focal cross-entropy loss between true labels and predictions.

  Binary cross-entropy loss is often used for binary (0 or 1) classification
  tasks. The loss function requires the following inputs:

  - `y_true` (true label): This is either 0 or 1.
  - `y_pred` (predicted value): This is the model's prediction, i.e, a single
    floating-point value which either represents a
    [logit](https://en.wikipedia.org/wiki/Logit), (i.e, value in [-inf, inf]
    when `from_logits=True`) or a probability (i.e, value in `[0., 1.]` when
    `from_logits=False`).

  According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
  helps to apply a "focal factor" to down-weight easy examples and focus more on
  hard examples. By default, the focal tensor is computed as follows:

  `focal_factor = (1 - output) ** gamma` for class 1
  `focal_factor = output ** gamma` for class 0
  where `gamma` is a focusing parameter. When `gamma=0`, this function is
  equivalent to the binary crossentropy loss.

  With the `compile()` API:

  ```python
  model.compile(
    loss=tf.keras.losses.BinaryFocalCrossentropy(gamma=2.0, from_logits=True),
    ....
  )
  ```

  As a standalone function:

  >>> # Example 1: (batch_size = 1, number of samples = 4)
  >>> y_true = [0, 1, 0, 0]
  >>> y_pred = [-18.6, 0.51, 2.94, -12.8]
  >>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=2, from_logits=True)
  >>> loss(y_true, y_pred).numpy()
  0.691

  >>> # Example 2: (batch_size = 2, number of samples = 4)
  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[-18.6, 0.51], [2.94, -12.8]]
  >>> # Using default 'auto'/'sum_over_batch_size' reduction type.
  >>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=3, from_logits=True)
  >>> loss(y_true, y_pred).numpy()
  0.647

  >>> # Using 'sample_weight' attribute
  >>> loss(y_true, y_pred, sample_weight=[0.8, 0.2]).numpy()
  0.133

  >>> # Using 'sum' reduction` type.
  >>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=4, from_logits=True,
  ...     reduction=tf.keras.losses.Reduction.SUM)
  >>> loss(y_true, y_pred).numpy()
  1.222

  >>> # Using 'none' reduction type.
  >>> loss = tf.keras.losses.BinaryFocalCrossentropy(gamma=5, from_logits=True,
  ...     reduction=tf.keras.losses.Reduction.NONE)
  >>> loss(y_true, y_pred).numpy()
  array([0.0017 1.1561], dtype=float32)

  Args:
    gamma: A focusing parameter used to compute the focal factor, default is
      `2.0` as mentioned in the reference
      [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
    from_logits: Whether to interpret `y_pred` as a tensor of
      [logit](https://en.wikipedia.org/wiki/Logit) values. By default, we
      assume that `y_pred` are probabilities (i.e., values in `[0, 1]`).
    label_smoothing: Float in `[0, 1]`. When `0`, no smoothing occurs. When >
      `0`, we compute the loss between the predicted labels and a smoothed
      version of the true labels, where the smoothing squeezes the labels
      towards `0.5`. Larger values of `label_smoothing` correspond to heavier
      smoothing.
    axis: The axis along which to compute crossentropy (the features axis).
      Defaults to `-1`.
    reduction: Type of `tf.keras.losses.Reduction` to apply to
      loss. Default value is `AUTO`. `AUTO` indicates that the reduction
      option will be determined by the usage context. For almost all cases
      this defaults to `SUM_OVER_BATCH_SIZE`. When used with
      `tf.distribute.Strategy`, outside of built-in training loops such as
      `tf.keras`, `compile()` and `fit()`, using `SUM_OVER_BATCH_SIZE` or
      `AUTO` will raise an error. Please see this custom training [tutorial](
      https://www.tensorflow.org/tutorials/distribute/custom_training) for
      more details.
    name: Name for the op. Defaults to 'binary_focal_crossentropy'.
  """

  def __init__(
      self,
      gamma=2.0,
      from_logits=False,
      label_smoothing=0.,
      axis=-1,
      reduction=losses_utils.ReductionV2.AUTO,
      name='binary_focal_crossentropy',
  ):
    """Initializes `BinaryFocalCrossentropy` instance."""
    super().__init__(
        binary_focal_crossentropy,
        gamma=gamma,
        name=name,
        reduction=reduction,
        from_logits=from_logits,
        label_smoothing=label_smoothing,
        axis=axis)
    self.from_logits = from_logits
    self.gamma = gamma

  def get_config(self):
    config = {
        'gamma': self.gamma,
    }
    base_config = super(BinaryFocalCrossentropy, self).get_config()
    return dict(list(base_config.items()) + list(config.items()))


@keras_export('keras.metrics.binary_focal_crossentropy',
              'keras.losses.binary_focal_crossentropy')
@tf.__internal__.dispatch.add_dispatch_support
def binary_focal_crossentropy(
    y_true,
    y_pred,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.,
    axis=-1,
):
  """Computes the binary focal crossentropy loss.

  According to [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf), it
  helps to apply a focal factor to down-weight easy examples and focus more on
  hard examples. By default, the focal tensor is computed as follows:

  `focal_factor = (1 - output)**gamma` for class 1
  `focal_factor = output**gamma` for class 0
  where `gamma` is a focusing parameter. When `gamma` = 0, this function is
  equivalent to the binary crossentropy loss.

  Standalone usage:

  >>> y_true = [[0, 1], [0, 0]]
  >>> y_pred = [[0.6, 0.4], [0.4, 0.6]]
  >>> loss = tf.keras.losses.binary_focal_crossentropy(y_true, y_pred, gamma=2)
  >>> assert loss.shape == (2,)
  >>> loss.numpy()
  array([0.330, 0.206], dtype=float32)

  Args:
    y_true: Ground truth values, of shape `(batch_size, d0, .. dN)`.
    y_pred: The predicted values, of shape `(batch_size, d0, .. dN)`.
    gamma: A focusing parameter, default is `2.0` as mentioned in the reference.
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in `[0, 1]`. If higher than 0 then smooth the labels
      by squeezing them towards `0.5`, i.e., using `1. - 0.5 * label_smoothing`
      for the target class and `0.5 * label_smoothing` for the non-target class.
    axis: The axis along which the mean is computed. Defaults to `-1`.

  Returns:
    Binary focal crossentropy loss value. shape = `[batch_size, d0, .. dN-1]`.
  """
  y_pred = tf.convert_to_tensor(y_pred)
  y_true = tf.cast(y_true, y_pred.dtype)
  label_smoothing = tf.convert_to_tensor(label_smoothing, dtype=y_pred.dtype)

  def _smooth_labels():
    return y_true * (1.0 - label_smoothing) + 0.5 * label_smoothing

  y_true = tf.__internal__.smart_cond.smart_cond(label_smoothing,
                                                 _smooth_labels, lambda: y_true)

  return backend.mean(
      backend.binary_focal_crossentropy(
          target=y_true,
          output=y_pred,
          gamma=gamma,
          from_logits=from_logits,
      ),
      axis=axis,
  )


@dispatch.dispatch_for_types(binary_focal_crossentropy, tf.RaggedTensor)
def _ragged_tensor_binary_focal_crossentropy(
    y_true,
    y_pred,
    gamma=2.0,
    from_logits=False,
    label_smoothing=0.,
    axis=-1,
):
  """Implements support for handling RaggedTensors.

  Expected shape: `(batch, sequence_len)` with sequence_len being variable per
  batch.
  Return shape: `(batch,)`; returns the per batch mean of the loss values.

  When used by BinaryFocalCrossentropy() with the default reduction
  (SUM_OVER_BATCH_SIZE), the reduction averages the per batch losses over
  the number of batches.

  Args:
    y_true: Tensor of one-hot true targets.
    y_pred: Tensor of predicted targets.
    gamma: A focusing parameter, default is `2.0` as mentioned in the reference
      [Lin et al., 2018](https://arxiv.org/pdf/1708.02002.pdf).
    from_logits: Whether `y_pred` is expected to be a logits tensor. By default,
      we assume that `y_pred` encodes a probability distribution.
    label_smoothing: Float in `[0, 1]`. If > `0` then smooth the labels. For
      example, if `0.1`, use `0.1 / num_classes` for non-target labels
      and `0.9 + 0.1 / num_classes` for target labels.
    axis: Axis along which to compute crossentropy.

  Returns:
    Binary focal crossentropy loss value.
  """
  fn = functools.partial(
      binary_focal_crossentropy,
      gamma=gamma,
      from_logits=from_logits,
      label_smoothing=label_smoothing,
      axis=axis,
  )
  return _ragged_tensor_apply_loss(fn, y_true, y_pred)

def compare(h1, h2, what='crystal'):
    pylab.figure(1)
    for key in h1:
        if what in key and 'loss' not in key:
            pylab.plot(h1[key], label=key)
    pylab.legend()
    pylab.figure(2)
    for key in h2:
        if what in key and 'loss' not in key:
            pylab.plot(h2[key], label=key)
    pylab.legend()
    pylab.show()


def plot_history(history, h=None, notions=['crystal', 'loop_inside', 'loop', 'stem', 'pin', 'capillary', 'ice', 'foreground']):
    if h is None:
        h = pickle.load(open(history, 'rb'), encoding='bytes')
    template = history.replace('.history', '')
    pylab.figure(figsize=(16, 9))
    pylab.title(template)
    for notion in notions:
        key = 'val_%s_BIoU_1' % notion
        if key in h:
            pylab.plot(h[key], 'o-', label=notion)
        else:
            continue
    pylab.ylim([-0.1, 1.1])
    pylab.grid(True)
    pylab.legend()
    pylab.savefig('%s_metrics.png' % template)
    
def analyse_histories(notions=['crystal', 'loop_inside', 'loop', 'stem', 'pin', 'foreground']):
    histories = glob.glob('*.history') + glob.glob('experiments/*.history') + glob.glob('backup/*.history')
    metrics_table = {}
    for history in histories:
        print(history)
        h = pickle.load(open(history, 'rb'), encoding='bytes')
        plot_history(history, h=h, notions=notions)
        val_metrics = []
        for notion in notions:
            key = 'val_%s_BIoU_1' % notion
            if key in h:
                val_metrics.append(h['val_%s_BIoU_1' % notion])
        val_metrics = np.array(val_metrics)
        try:
            best = val_metrics.max(axis=1).T
            best
        except:
            best = 'problem in determining expected metrics'
        
        line = '%s: %s' % (best, history)
        print(line)
        os.system('echo "%s" >> histories.txt' % line)
    
def resize_images(images, size, method='bilinear', align_corners=False):
    """ See https://www.tensorflow.org/versions/master/api_docs/python/tf/image/resize_images .

    Args
        method: The method used for interpolation. One of ('bilinear', 'nearest', 'bicubic', 'area').
    """
    methods = {
        'bilinear': tf.image.ResizeMethod.BILINEAR,
        'nearest': tf.image.ResizeMethod.NEAREST_NEIGHBOR,
        'bicubic': tf.image.ResizeMethod.BICUBIC,
        'area': tf.image.ResizeMethod.AREA,
    }
    return tf.image.resize_images(images, size, methods[method], align_corners)

class UpsampleLike(keras.layers.Layer):
    """
    Keras layer for upsampling a Tensor to be the same shape as another Tensor.
    based on https://github.com/xuannianz/keras-fcos.git
    """

    def call(self, inputs, **kwargs):
        source, target = inputs
        target_shape = K.shape(target)
        return resize_images(source, (target_shape[1], target_shape[2]), method='nearest')

    def compute_output_shape(self, input_shape):
        return (input_shape[0][0],) + input_shape[1][1:3] + (input_shape[0][-1],)

def get_pixels(directory='/nfs/data2/Martin/Research/murko/images_and_labels', notions=['crystal', 'loop_inside', 'loop', 'stem', 'pin', 'capillary', 'ice', 'foreground'], print_table=True):
    masks = glob.glob('%s/*/masks.npy' % directory)
    pixel_counts = dict([(notion, 0) for notion in notions])
    pixel_counts['total'] = 0
    for mask in masks:
        m = np.load(mask)
        for k, notion in enumerate(notions):
            pixel_counts[notion] += m[:,:,k].sum()
        pixel_counts['total'] += np.prod(m.shape[:2])
    if print_table:
        print('total pixels %d (%.3fG)'.ljust(15) % (pixel_counts['total'], pixel_counts['total']/1e9))
        print('total foreground %d (%.3fG, %.4f of all)'.ljust(15) % (pixel_counts['foreground'], pixel_counts['foreground']/1e9, pixel_counts['foreground']/pixel_counts['total']))
        print()
        print('notion'.rjust(15), 'fraction_label'.rjust(15), 'fraction_total'.rjust(15), 'weight_label'.rjust(20), 'weight_total'.rjust(20))
        for key in pixel_counts:
            print(key.rjust(15), '%.4f'.rjust(10) % (pixel_counts[key]/pixel_counts['foreground']), '%.4f'.rjust(15) % (pixel_counts[key]/pixel_counts['total']), '%3.1f'.zfill(2).rjust(20) % (pixel_counts['foreground']/pixel_counts[key]), '%3.1f'.zfill(2).rjust(20) % (pixel_counts['total']/pixel_counts[key]))
    return pixel_counts
    
def get_flops(model_h5_path):
    session = tf.compat.v1.Session()
    graph = tf.compat.v1.get_default_graph()
        

    with graph.as_default():
        with session.as_default():
            model = tf.keras.models.load_model(model_h5_path)

            run_meta = tf.compat.v1.RunMetadata()
            opts = tf.compat.v1.profiler.ProfileOptionBuilder.float_operation()
        
            # We use the Keras session graph in the call to the profiler.
            flops = tf.compat.v1.profiler.profile(graph=graph,
                                                  run_meta=run_meta, cmd='op', options=opts)
        
            return flops.total_float_ops


class WSConv2D(tf.keras.layers.Conv2D):
    ''' https://github.com/joe-siyuan-qiao/WeightStandardization '''
    def __init__(self, *args, **kwargs):
        super(WSConv2D, self).__init__(*args, **kwargs)
        self.eps = 1.e-5 
        self.std = False
        
    def standardize_kernel(self, kernel):

        original_dtype = kernel.dtype
        
        mean = tf.math.reduce_mean(kernel, axis=(0, 1, 2), keepdims=True)
        kernel = kernel - mean

        if self.std:
            std = tf.keras.backend.std(kernel, axis=[0, 1, 2], keepdims=True)
            std = std + tf.constant(self.eps, dtype=std.dtype)
            kernel = kernel / std 
            
        kernel = tf.cast(kernel, dtype=original_dtype)
        
        return kernel 

    def call(self, inputs):
        self.kernel.assign(self.standardize_kernel(self.kernel))
        return super().call(inputs)
    
class WSSeparableConv2D(tf.keras.layers.SeparableConv2D):
    ''' https://github.com/joe-siyuan-qiao/WeightStandardization '''
    def __init__(self, *args, **kwargs):
        super(WSSeparableConv2D, self).__init__(*args, **kwargs)
        self.eps = 1.e-5 
        self.std = False
        
    def standardize_kernel(self, kernel):

        original_dtype = kernel.dtype
        
        mean = tf.math.reduce_mean(kernel, axis=(0, 1, 2), keepdims=True)
        kernel = kernel - mean
        
        if self.std:
            std =  tf.math.reduce_std(kernel, axis=[0, 1, 2], keepdims=True)
            std = std + tf.constant(self.eps, dtype=std.dtype)
            kernel = kernel / std 
            
        kernel = tf.cast(kernel, dtype=original_dtype)
        
        return kernel 

    def call(self, inputs):
        self.pointwise_kernel.assign(self.standardize_kernel(self.pointwise_kernel))
        self.depthwise_kernel.assign(self.standardize_kernel(self.depthwise_kernel))
        
        return super().call(inputs)

def generate_click_loss_and_metric_figures(click_radius=360e-3, image_shape=(1024, 1360), nclicks=10, ntries=1000, display=True):
    resize_factor = np.array(image_shape)/np.array((1024, 1360))
    distances = []
    bfcs = []
    bio1 = []
    bio1m = tf.keras.metrics.BinaryIoUm(target_class_ids=[1], threshold=0.5)
    bio0 = []
    bio0m = tf.keras.metrics.BinaryIoUm(target_class_ids=[0], threshold=0.5)
    biob = []
    biobm = tf.keras.metrics.BinaryIoUm(target_class_ids=[0, 1], threshold=0.5)
    concepts = {'bfcs': bfcs,
                'bio1': bio1,
                'bio0': bio0,
                'biob': biob,
                'distances': distances}

    for k in range(nclicks):
        click = (np.array(image_shape) * np.random.rand(2,)).astype(int)
        cpi_true = click_probability_image(click[1], click[0], image_shape, click_radius=click_radius, resize_factor=resize_factor, scale_click=False)
        cpi_true = np.expand_dims(cpi_true, (0, -1))
        for n in range(ntries//nclicks):
            tclick = (np.array(image_shape) * np.random.rand(2,)).astype(int)
            cpi_pred = click_probability_image(tclick[1], tclick[0], image_shape, click_radius=click_radius, resize_factor=resize_factor, scale_click=False)
            cpi_pred = np.expand_dims(cpi_pred, (0, -1))
            concepts['distances'].append(np.linalg.norm(click-tclick, 2))
            concepts['bfcs'].append(tf.keras.losses.binary_focal_crossentropy(cpi_true, cpi_pred).numpy().mean())
            bio1m.reset_state()
            bio1m.update_state(cpi_true, cpi_pred)
            concepts['bio1'].append(bio1m.result().numpy())
            bio0m.reset_state()
            bio0m.update_state(cpi_true, cpi_pred)
            concepts['bio0'].append(bio0m.result().numpy())
            biobm.reset_state()
            biobm.update_state(cpi_true, cpi_pred)
            concepts['biob'].append(biobm.result().numpy())

    for concept in concepts:
        concepts[concept] = np.array(concepts[concept])
    concepts['distances'] /= np.linalg.norm(image_shape, 2)
    concepts['bfcs'] /= concepts['bfcs'].max()
    pylab.figure(figsize=(16, 9))
    pylab.title('image shape %dx%d, click_radius=%.3f' % (image_shape[0], image_shape[1], click_radius))
    for concept in ['bfcs', 'bio1', 'bio0', 'biob']:
        pylab.plot(concepts['distances'], concepts[concept], 'o', label=concept)
    pylab.xlabel('distances')
    pylab.ylabel('loss/metrics')
    pylab.savefig('click_metric_cr_%.3f_img_shape_%dx%d.png' % (click_radius, image_shape[0], image_shape[1]))
    pylab.legend()
    if display:
        pylab.show()
    return concepts
    
class ClickMetric(tf.keras.metrics.MeanAbsoluteError):

    def __init__(self,
                 name='click_metric',
                dtype=None):

        super(ClickMetric, self).__init__(
            name=name,
            dtype=dtype
        )

    def update_state(self, y_true, y_pred, sample_weight=None):
        return super().update_state(y_true, y_pred, sample_weight)

 
class ClickLoss(tf.keras.losses.MeanSquaredError):
    
    def call(self, ci_true, ci_pred):

        com_true = tf_center_of_mass(ci_true)
        com_pred = tf_centre_of_mass(ci_pred)

        mse = super().call(com_true, com_pred)
        mse = replacenan(mse)
        bcl = tf.reduce_mean(tf.keras.losses.binary_crossentropy(ci_true, ci_pred), axis=(1, 2))
        click_present = tf.reshape(K.max(ci_true, axis=(1, 2)), (-1))
        total = bcl*(1-click_present) + mse*(click_present)
        
        return total

def gauss2d(x=0, y=0, mx=0, my=0, sx=1, sy=1):
    return np.exp(-((x - mx)**2. / (2. * sx**2.) + (y - my)**2. / (2. * sy**2.)))

def click_probability_image(click_x, click_y, img_shape, zoom=1, click_radius=320e-3, resize_factor=1., scale_click=True):
    x = np.arange(0, img_shape[1], 1)
    y = np.arange(0, img_shape[0], 1)
    x, y = np.meshgrid(x, y)
    if scale_click:
        mmppx = calibrations[zoom]/resize_factor
    else:
        mmppx = calibrations[1]/resize_factor
    sx = click_radius/mmppx.mean()
    sy = sx
    z = gauss2d(x, y, mx=click_x, my=click_y, sx=sx, sy=sy)
    return z 

def replacenan(t):
    return tf.where(tf.math.is_nan(t), tf.zeros_like(t), t)
    
def click_loss(ci_true, ci_pred):
    total = tf.keras.losses.mean_squared_error(ci_true, ci_pred)
    return total

def tf_center_of_mass(image_batch, threshold=0.5):
    '''https://stackoverflow.com/questions/51724450/finding-centre-of-mass-of-tensor-tensorflow '''
    print(image_batch.shape)
    tf.cast(image_batch >= threshold, tf.float32)
    batch_size, height, width, depth = image_batch.shape
    # Make array of coordinates (each row contains three coordinates)
    
    ii, jj, kk = tf.meshgrid(tf.range(height), tf.range(width), tf.range(depth), indexing='ij')
    coords = tf.stack([tf.reshape(ii, (-1,)), tf.reshape(jj, (-1,)), tf.reshape(kk, (-1))], axis=-1)
    coords = tf.cast(coords, tf.float32)
    # Rearrange input into one vector per volume
    volumes_flat = tf.reshape(image_batch, [-1, height*width, 1])
    # Compute total mass for each volume
    total_mass = tf.reduce_sum(volumes_flat, axis=1)
    # Compute centre of mass
    centre_of_mass = tf.reduce_sum(volumes_flat * coords, axis=1) / total_mass
    
    return centre_of_mass

def click_image_loss(ci_true, ci_pred):
    if K.max(ci_true) == 0:
        return tf.keras.losses.binary_crossentropy(y_true, y_pred)
    y_true = centre_of_mass(ci_true)
    y_pred = centre_of_mass(ci_pred)
    return tf.keras.losses.mean_squared_error(y_true, y_pred)

def click_mean_absolute_error(ci_true, ci_pred):
    if K.max(ci_pred) < 0.5 and K.max(ci_true) == 0:
        return 0
    
    y_true = centre_of_mass(ci_true)
    y_pred = centre_of_mass(ci_pred)
    return tf.keras.losses.mean_absolute_error(y_true, y_pred)

def click_batch_loss(click_true_batch, click_pred_image_batch):
    return [click_loss(click_true, click_pred_image) for click_true, click_pred_image in zip(click_true_batch, click_pred_image_batch)]

def get_click_from_single_click_image(click_image):
    click_pred = np.zeros((3,))
    m = click_image.max()
    click_pred[:2] = np.array(np.unravel_index(np.argmax(click_image), click_image.shape)[:2], dtype='float32')
    click_pred[2] = m
    return click_pred

def get_clicks_from_click_image_batch(click_image_batch):
    input_shape = click_image_batch.shape
    output_shape = (input_shape[0], 3)
    click_pred = np.zeros(output_shape)
    for k, click_image in enumerate(click_image_batch):
        click_pred[k] = get_click_from_single_click_image(click_image)
    return click_pred

def mean_iou(y_true, y_pred):
    prec = []
    for t in np.arange(0.5, 1.0, 0.05):
        y_pred_ = tf.to_int32(y_pred > t)
        score, up_opt = tf.metrics.mean_iou(y_true, y_pred_, 2)
        K.get_session().run(tf.local_variables_initializer())
        with tf.control_dependencies([up_opt]):
            score = tf.identity(score)
        prec.append(score)
    return K.mean(K.stack(prec), axis=0)

def display_target(target_array):
    normalized_array = (target_array.astype("uint8")) * 127
    plt.axis("off")
    plt.imshow(normalized_array[:, :, 0])

def flip_axis(x, axis):
    x = np.asarray(x).swapaxes(axis, 0)
    x = x[::-1, ...]
    x = x.swapaxes(0, axis)
    return x

def get_transposed_img_and_target(img, target):
    new_axes_order = (1, 0,) + tuple(range(2, len(img.shape)))
    img = np.transpose(img, new_axes_order)
    new_axes_order = (1, 0,) + tuple(range(2, len(target.shape)))
    target = np.transpose(target, new_axes_order) #[:len(target.shape)])
    return img, target
    
def get_flipped_img_and_target(img, target):
    axis = random.choice([0, 1])
    img = flip_axis(img, axis)
    target = flip_axis(target, axis)
    return img, target

def get_transformed_img_and_target(img, target, default_transform_gang=[0, 0, 0, 0, 1, 1], zoom_factor=0.5, shift_factor=0.5, shear_factor=45, size=(512, 512), rotate_probability=1, shift_probability=1, shear_probability=1, zoom_probability=1, theta_min=-180., theta_max=180., resize=False, random_brightness=True, random_saturation=True):
    if resize:
        img = resize(img, size, anti_aliasing=True)
        target = resize(target, size, anti_aliasing=True)
    theta, tx, ty, shear, zx, zy = default_transform_gang
    size_y, size_x = img.shape[:2]
    # rotate
    if random.random() < rotate_probability:
        theta = random.uniform(theta_min, theta_max)
    # shear
    if random.random() < shear_probability:
        shear = random.uniform(-shear_factor, +shear_factor)
    # shift
    if random.random() < shift_probability:
        tx = random.uniform(-shift_factor*size_x, +shift_factor*size_x)
        ty = random.uniform(-shift_factor*size_y, +shift_factor*size_y)
    # zoom
    if random.random() < zoom_probability:
        zx = random.uniform(1-zoom_factor, 1+zoom_factor)
        zy = zx
    #brightness
    if random_brightness:
        img = image.random_brightness(img, [0.75, 1.25])/255.
    if random_saturation:
        img = image.random_channel_shift(img, 0.5, channel_axis=2)
        
    if np.any(np.array([theta, tx, ty, shear, zx, zy]) != default_transform_gang):
        transform_arguments = {'theta': theta, 'tx': tx, 'ty': ty, 'shear': shear, 'zx': zx, 'zy': zy}
        img = image.apply_affine_transform(img, **transform_arguments)
        target = image.apply_affine_transform(target, **transform_arguments)
        #target = image.apply_affine_transform(target, fill_mode='constant', cval=0, **transform_arguments)
    return img, target

def get_dataset(batch_size, img_size, img_paths, augment=False):
    dataset = tf.data.Dataset.from_tensor_slices(img_paths)
    
def size_differs(original_size, img_size):
    return original_size[0] != img_size[0] or original_size[1] != img_size[1]

def augment_sample(img_path, img, target, user_click, do_swap_backgrounds, do_flip, do_transpose, zoom, candidate_backgrounds, notions, zoom_factor, shift_factor, shear_factor):
    if do_swap_backgrounds is True and 'background' not in img_path:
        new_background = random.choice(candidate_backgrounds[zoom])
        if size_differs(img.shape[:2], new_background.shape[:2]):
            new_background = resize(new_background, img.shape[:2], anti_aliasing=True)
        img[target[:,:,notions.index('foreground')]==0] = new_background[target[:,:,notions.index('foreground')]==0]
            
    if self.augment and do_transpose is True:
        img, target = get_transposed_img_and_target(img, target)
    
    if self.augment and do_flip is True:
        img, target = get_flipped_img_and_target(img, target)
        
    if self.augment:
        img, target = get_transformed_img_and_target(img, target, zoom_factor=zoom_factor, shift_factor=shift_factor, shear_factor=shear_factor)

    return img, target

def get_img_and_target(img_path, img_string='img.jpg', label_string='masks.npy'):
    original_image = load_img(img_path)
    original_size = original_image.size[::-1]
    img = img_to_array(original_image, dtype="float32")/255.
    masks_name = img_path.replace(img_string, label_string)
    target = np.load(masks_name)
    return img, target

def get_img(img_path, size=(224, 224)):
    original_image = load_img(img_path)
    img = img_to_array(original_image, dtype="float32")/255.    
    img = resize(img, size, anti_aliasing=True)
    return img

def get_batch(i, img_paths, batch_size):
    half, r = divmod(batch_size, 2)
    indices = np.arange(i-half, i+half+r)
    return [img_paths[divmod(item, len(img_paths))[1]] for item in indices]

def load_ground_truth_image(path, target_size):
    ground_truth = np.expand_dims(load_img(path, target_size=target_size, color_mode="grayscale"), 2)
    if ground_truth.max() > 0:
        ground_truth = np.array(ground_truth/ground_truth.max(), dtype='uint8')
    else:
        ground_truth = np.array(ground_truth, dtype='uint8')
    return ground_truth
    
class MultiTargetDataset(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, img_paths, img_string='img.jpg', label_string='masks.npy', click_string='user_click.npy', augment=False, transpose=True, flip=True, swap_backgrounds=True, zoom_factor=0.5, shift_factor=0.5, shear_factor=45, default_transform_gang=[0, 0, 0, 0, 1, 1], scale_click=False, click_radius=320e-3, min_scale=0.15, max_scale=1.0, dynamic_batch_size=False, number_batch_size_scales=32, possible_ratios=[0.75, 1.], pixel_budget=768*992, artificial_size_increase=1, notions=['crystal', 'loop_inside', 'loop', 'stem', 'pin', 'capillary', 'ice', 'foreground', 'click'], notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'capillary': 5, 'ice': 6, 'foreground': 7, 'click':-1}, shuffle_at_0=False, click='segmentation', target=True, black_and_wight=False):
        self.batch_size = batch_size
        self.img_size = img_size
        if artificial_size_increase > 1:
           self.img_paths = img_paths*int(artificial_size_increase) 
        else:
            self.img_paths = img_paths
        self.nimages = len(self.img_paths)
        self.img_string = img_string
        self.label_string = label_string
        self.click_string = click_string
        self.augment = augment
        self.transpose = transpose
        self.flip = flip
        self.swap_backgrounds = swap_backgrounds
        self.zoom_factor = zoom_factor
        self.shift_factor = shift_factor
        self.shear_factor = shear_factor
        self.default_transform_gang = np.array(default_transform_gang)
        self.scale_click = scale_click
        self.click_radius = click_radius
        self.dynamic_batch_size = dynamic_batch_size
        if self.dynamic_batch_size:
            self.batch_size = 1
        self.possible_scales = np.linspace(min_scale, max_scale, number_batch_size_scales)
        self.possible_ratios = possible_ratios
        self.pixel_budget = pixel_budget
        self.notions = notions
        self.notion_indices = notion_indices
        self.candidate_backgrounds = {}
        self.batch_img_paths = []
        if self.swap_backgrounds:
            backgrounds = glob.glob('./Backgrounds/*.jpg') + glob.glob('./Backgrounds/*.tif')
            for img_path in backgrounds:
                zoom = int(re.findall('.*_zoom_([\d]*).*', img_path)[0])
                background = load_img(img_path)
                background = img_to_array(background, dtype="float32")/255.
                if zoom in self.candidate_backgrounds:
                    self.candidate_backgrounds[zoom].append(background)
                else:
                    self.candidate_backgrounds[zoom] = [background]
        self.shuffle_at_0 = shuffle_at_0
        self.click = click
        self.target = target
        self.black_and_wight = black_and_wight
        
    def __len__(self):
        return math.ceil(len(self.img_paths)/self.batch_size)
    
    def consider_click(self):
        if self.target and 'click' in self.notions:
            return True
        return False
    
    def __getitem__(self, idx):
        if idx == 0 and self.shuffle_at_0:
            random.Random().shuffle(self.img_paths)
        if self.dynamic_batch_size:
            img_size = get_img_size_as_scale_of_pixel_budget(random.choice(self.possible_scales), pixel_budget=self.pixel_budget, ratio=random.choice(self.possible_ratios))
            batch_size = get_dynamic_batch_size(img_size, pixel_budget=self.pixel_budget)
            i = idx
            self.batch_img_paths = get_batch(i, self.img_paths, batch_size)
        else:
            img_size = self.img_size[:]
            batch_size = self.batch_size
            i = idx * self.batch_size
            start_index = i
            end_index = i + batch_size
            self.batch_img_paths = self.img_paths[start_index: end_index]
            
        final_img_size = img_size[:]
        batch_size = len(self.batch_img_paths) # this handles case at the very last step ...
        
        do_flip = False
        do_transpose = False
        do_transform = False
        do_swap_backgrounds = False
        do_black_and_wight = False
            
        if self.augment:
            if self.transpose and random.random() > 0.5:
                final_img_size = img_size[::-1]
                do_transpose = True
            if self.flip and random.random() > 0.5:
                do_flip = True
            if self.augment and random.random() > 0.1:
                do_transform = True
            if self.swap_backgrounds and random.random() > 0.25:
                do_swap_backgrounds = True
            if self.black_and_wight and random.random() > 0.5:
                do_black_and_wight = True
                
        x = np.zeros((batch_size,) + final_img_size + (3,), dtype="float32")
        y = [np.zeros((batch_size,) + final_img_size + (1,), dtype="uint8") for notion in self.notions if 'click' not in notion]
        if 'click' in self.notions:
            if click == 'click_segmentation':
                y.append(np.zeros((batch_size,) + final_img_size + (1,), dtype="float32"))
            else:
                y.append(np.zeros((batch_size,) + (3,), dtype="float32"))
            
        for j, img_path in enumerate(self.batch_img_paths):
            resize_factor = 1.
            original_image = load_img(img_path)
            original_size = original_image.size[::-1]
            img = img_to_array(original_image, dtype="float32")/255.
            masks_name = img_path.replace(self.img_string, self.label_string)
            user_click_name = img_path.replace(self.img_string, self.click_string)
            if self.target:
                target = np.load(masks_name)
            try:
                zoom = int(re.findall('.*_zoom_([\d]*).*', img_path)[0])
            except:
                zoom = 1
          
            if size_differs(original_size, img_size):
                resize_factor = np.array(img_size)/np.array(original_size)
                
            if self.consider_click():
                user_click = np.array(np.load(user_click_name)).astype('float32')
                click_present = all(user_click[:2]>=0)
                if self.augment and click_present:
                    click_mask = get_cpi_from_user_click(user_click, target.shape[:2], 1., img_path, click_radius=self.click_radius, zoom=zoom, scale_click=self.scale_click)
                    target = np.concatenate([target, click_mask], axis=2)
                elif click_present:
                    user_click_frac = user_click * resize_factor
                else:
                    user_click_frac = np.array([np.nan, np.nan])
                    
            if self.target and np.all(target[:,:,self.notions.index('foreground')] == 0):
                do_swap_backgrounds = False
                
            if self.augment and do_transpose is True:
                img, target = get_transposed_img_and_target(img, target)
            
            if self.augment and do_flip is True:
                img, target = get_flipped_img_and_target(img, target)
                
            if do_transform is True:
                img, target = get_transformed_img_and_target(img, target, zoom_factor=self.zoom_factor, shift_factor=self.shift_factor, shear_factor=self.shear_factor)

            if self.augment and do_swap_backgrounds is True and 'background' not in img_path:
                new_background = random.choice(self.candidate_backgrounds[zoom])
                if size_differs(img.shape[:2], new_background.shape[:2]):
                    new_background = resize(new_background, img.shape[:2], anti_aliasing=True)
                img[target[:,:,self.notions.index('foreground')]==0] = new_background[target[:,:,self.notions.index('foreground')]==0]
                
            if size_differs(img.shape[:2], final_img_size):
                img = resize(img, final_img_size, anti_aliasing=True)
                if self.target:
                    target = resize(target.astype('float32'), final_img_size, mode='constant', cval=0, anti_aliasing=False, preserve_range=True)
                
            if self.augment and self.consider_click() and click_present:
                transformed_click = target[:,:,-1]
                user_click = np.unravel_index(np.argmax(transformed_click), transformed_click.shape)[:2]
                user_click_frac = np.array(user_click)/np.array(final_img_size)
                if self.click == 'click_segmentation':
                    cpi = get_cpi_from_user_click(user_click, final_img_size, resize_factor, img_path + 'augment', click_radius=self.click_radius, zoom=zoom, scale_click=self.scale_click)
                    target[:,:,-1] = cpi[:,:,0]

            if self.consider_click() and self.click=='click_segmentation':
                target[:,:,:-1] = (target[:,:,:-1]>0).astype('uint8')
            elif self.consider_click() and self.click=='click_regression':
                click_present = int(click_present)
                y_click, x_click = user_click_frac
            elif self.target:
                target = (target>0).astype('uint8')
            if do_black_and_wight:
                img_bw = img.mean(axis=2)
                img = np.expand_dims(img_bw, axis=2)
            x[j] = img
            if self.target:
                for k, notion in enumerate(self.notions):
                    l = self.notion_indices[notion]
                    if l != -1:
                        y[k][j] = np.expand_dims(target[:,:,l], axis=2)
                    elif l == -1 and self.click == 'click_segmentation':
                        y[k][j] = np.expand_dims(target[:,:,l], axis=2)
                    elif l == -1 and self.click == 'click_regression':
                        y[k][j] = np.array([click_present, y_click, x_click])
                
        if self.target and len(y) == 1:
            y = y[0]
        return x, y
                
class SampleSegmentationDataset(keras.utils.Sequence):
    """Helper to iterate over the data (as Numpy arrays)."""

    def __init__(self, batch_size, img_size, input_img_paths, img_string='img.jpg', label_string='foreground.png', augment=False):
        self.batch_size = batch_size
        self.img_size = img_size
        self.input_img_paths = input_img_paths
        self.img_string = img_string
        self.label_string = label_string
        self.augment = augment
        self.zoom_factor = 0.2
        self.shift_factor = 0.25
        self.shear_factor = 15
        self.default_transform_gang = np.array([0, 0, 0, 0, 1, 1])
        
    def __len__(self):
        return len(self.input_img_paths) // self.batch_size

    def __getitem__(self, idx):
        """Returns tuple (input, target) correspond to batch #idx."""
        i = idx * self.batch_size
        self.batch_img_paths = self.input_img_paths[i : i + self.batch_size]
        batch_target_paths = [path.replace(self.img_string, self.label_string) for path in self.batch_img_paths]
        x = np.zeros((self.batch_size,) + self.img_size + (3,), dtype="float32")
        y = np.zeros((self.batch_size,) + self.img_size + (1,), dtype="uint8")
        for j, (img_path, target_path) in enumerate(zip(batch_input_img_paths, batch_target_img_paths)):
            img = img_to_array(load_img(img_path, target_size=self.img_size), dtype="float32")/255.
            target = load_ground_truth_image(target_path, target_size=self.img_size)

            if self.augment:
                 img, target = get_transformed_img_and_target(img, target, zoom_factor=self.zoom_factor, shift_factor=self.shift_factor, shear_factor=self.shear_factor)
            x[j] = img
            y[j] = target 
        return x, y
  
class CrystalClickDataset(keras.utils.Sequence):
    def __init__(self, batch_size, img_size, img_paths, augment=False, transpose=True, flip=True, zoom_factor=0.2, shift_factor=0.25, shear_factor=15, default_transform_gang=[0, 0, 0, 0, 1, 1], scale_click=False, click_radius=320e-3, min_scale=0.15, max_scale=1.0, dynamic_batch_size=True, number_batch_size_scales=32, possible_ratios=[0.75, 1.], pixel_budget=768*992):
        
        self.batch_size = batch_size
        self.img_size = img_size
        self.img_paths = img_paths
        self.augment = augment
        self.transpose = transpose
        self.flip = flip
        self.zoom_factor = zoom_factor
        self.shift_factor = shift_factor
        self.shear_factor = shear_factor
        self.default_transform_gang = np.array(default_transform_gang)
        self.scale_click = scale_click
        self.click_radius = click_radius
        self.dynamic_batch_size = dynamic_batch_size
        self.possible_scales = np.linspace(min_scale, max_scale, number_batch_size_scales)
        self.possible_ratios = possible_ratios
        self.pixel_budget = pixel_budget
        
        if self.dynamic_batch_size:
            self.batch_size = 1
            
    def __len__(self):
        return math.ceil(len(self.img_paths)/self.batch_size)
    
    def __getitem__(self, idx):
        if idx == 0:
            random.Random().shuffle(self.img_paths)
        if self.dynamic_batch_size:
            img_size = get_img_size_as_scale_of_pixel_budget(random.choice(self.possible_scales), pixel_budget=self.pixel_budget, ratio=random.choice(self.possible_ratios))
            batch_size = get_dynamic_batch_size(img_size, pixel_budget=self.pixel_budget)
            i = idx
        else:
            img_size = self.img_size
            batch_size = self.batch_size
            i = idx * self.batch_size
        
        final_img_size = img_size[:]
        batch_img_paths = self.img_paths[i: i+batch_size]
        batch_size = len(batch_img_paths) # this handles case at the very last step ...

        if self.augment:
            do_transpose = False
            if self.transpose and random.random() > 0.5:
                final_img_size = img_size[::-1]
                do_transpose = True
            do_flip = False
            if self.flip and random.random() > 0.5:
                do_flip = True
            
        x = np.zeros((batch_size,) + final_img_size + (3,), dtype="float32")
        y = np.zeros((batch_size,) + final_img_size + (1,), dtype="float32")
        for j, img_path in enumerate(batch_img_paths):
            user_click = None
            try:
                original_image = load_img(img_path)
                original_size = original_image.size[::-1]
                img = img_to_array(original_image, dtype="float32")/255.
                if np.any(np.isnan(img)):
                    os.system('echo this gave nan, please check %s >> click_generation_problems_new.txt' % img_path)
                    continue
                if original_size[0] > original_size[1]:
                    original_size = original_size[::-1]
                    img = np.reshape(img, original_size + img.shape[2:])
                    os.system('echo wrong ratio, please check %s >> click_generation_problems_new.txt' % img_path)
                img = resize(img, final_img_size)
            except:
                print(traceback.print_exc())
                os.system('echo load_img failed %s >> click_generation_problems_new.txt' % img_path)
                img = np.zeros(img_size + (3,))
                original_size = img_size[:]
                user_click = np.array([-1, -1])
            
            try:
                zoom = int(re.findall('.*_zoom_([\d]*).*', img_path)[0])
            except:
                zoom = 1
            if os.path.basename(img_path) == 'img.jpg' and user_click is None:
                user_click = np.load(img_path.replace('img.jpg', 'user_click.npy'))
            elif 'shapes_of_background' in img_path:
                user_click = np.array([-1., -1.])
            else:
                try:
                    user_click = np.array(list(map(float, re.findall('.*_y_([-\d]*)_x_([-\d]*).*', img_path)[0])))
                except:
                    user_click = np.array([-1., -1.])
                    
            resize_factor = np.array([1., 1.])
            if original_size[0] != img_size[0] and original_size[1] != img_size[1]:
                resize_factor = np.array(img_size)/np.array(original_size)
            
            user_click *= resize_factor
            user_click = user_click.astype('float32')
            cpi = get_cpi_from_user_click(user_click, final_img_size, resize_factor, img_path, click_radius=self.click_radius, zoom=zoom, scale_click=self.scale_click)
            if cpi is None:
                continue
            if self.augment:
                if do_transpose is True:
                    img, cpi = get_transposed_img_and_target(img, cpi)
                if do_flip is True:
                    img, cpi = get_flipped_img_and_target(img, cpi)
                img, cpi = get_transformed_img_and_target(img, cpi, zoom_factor=self.zoom_factor, shift_factor=self.shift_factor, shear_factor=self.shear_factor)
                
                if all(user_click[:2] >= 0):
                    user_click = np.unravel_index(np.argmax(cpi), cpi.shape)
                cpi = get_cpi_from_user_click(user_click[:2], final_img_size, resize_factor, img_path + 'augment', click_radius=self.click_radius, zoom=zoom, scale_click=self.scale_click)
            
            x[j] = img
            y[j] = cpi
                         
        return x, y
    
def get_cpi_from_user_click(user_click, img_size, resize_factor, img_path, click_radius=320e-3, zoom=1, scale_click=False):
    if all(np.array(user_click) >= 0):
        try:
            _y = int(user_click[0])
            _x = int(min(user_click[1], img_size[1]))
            if all(np.array((_y, _x)) >= 0):
                cpi = click_probability_image(_x, _y, img_size, click_radius=click_radius, zoom=zoom, resize_factor=resize_factor, scale_click=scale_click)
            else:
                cpi = np.zeros(img_size, dtype='float32')
        except:
            print(traceback.print_exc())
            os.system('echo %s >> click_generation_problems_new.txt' % img_path)
            return None
    else:
        cpi = np.zeros(img_size, dtype='float32')
    cpi = np.expand_dims(cpi, axis=2)
    return cpi
    
def get_data_augmentation():
    data_augmentation = keras.Sequential(
            [
                layers.RandomRotation(0.5),
                layers.RandomFlip(),
                layers.RandomZoom(0.2),
            ]
        )
    return data_augmentation
        
def get_resize_and_rescale(model_img_size):
    resize_and_rescale = keras.Sequential(
            [
                #layers.Resizing(model_img_size, model_img_size),
                layers.Rescaling(1./255)
            ]
        )
    return resize_and_rescale

def find_number_of_groups(c, g):
    w, r = divmod(c,g)
    if r == 0:
        return g
    else:
        return find_number_of_groups(c, g-1)
    
def get_kernel_regularizer(kernel_regularizer, weight_decay):
    if weight_decay == 0.:
        return None
    return getattr(regularizers, kernel_regularizer)(weight_decay)

def get_convolutional_layer(x, convolution_type, filters, filter_size=3, padding='same', activation='relu', kernel_initializer='he_normal', kernel_regularizer='l2', weight_decay=1e-4, use_bias=False, weight_standardization=True):
    if weight_standardization:
        if convolution_type == 'SeparableConv2D':
            x = WSSeparableConv2D(filters, filter_size, padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=get_kernel_regularizer(kernel_regularizer, weight_decay))(x)
        elif convolution_type == 'Conv2D':
            x = WSConv2D(filters, filter_size, padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=get_kernel_regularizer(kernel_regularizer, weight_decay))(x)
    else:
        x = getattr(layers, convolution_type)(filters, filter_size, padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=get_kernel_regularizer(kernel_regularizer, weight_decay))(x)
    return x
    

def get_tiramisu_layer(x, filters, filter_size=3, padding='same', activation='relu', convolution_type='Conv2D', kernel_initializer='he_normal', kernel_regularizer='l2', weight_decay=1e-4, use_bias=False,  normalization_type='GroupNormalization', bn_momentum=0.9, bn_epsilon=1.1e-5, gn_groups=16, dropout_rate=0.2, weight_standardization=True, invert=True):
    if invert:
        x = get_convolutional_layer(x, convolution_type, filters, filter_size, padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, weight_decay=weight_decay, weight_standardization=weight_standardization)
        x = get_normalization_layer(x, normalization_type, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, gn_groups=gn_groups)
        x = layers.Activation(activation=activation)(x)
    else:
        x = get_normalization_layer(x, normalization_type, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, gn_groups=gn_groups)
        x = layers.Activation(activation=activation)(x)
        x = get_convolutional_layer(x, convolution_type, filters, filter_size, padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, weight_decay=weight_decay, weight_standardization=weight_standardization)
    
    if dropout_rate:
        x = layers.Dropout(dropout_rate)(x)
    return x 

def get_dense_block(x, filters, number_of_layers, padding='same', activation='relu', convolution_type='Conv2D', dropout_rate=0.2, up=False, use_bias=False, kernel_initializer='he_normal', kernel_regularizer='l2', weight_decay=1e-4, bn_momentum=0.9, bn_epsilon=1.1e-5, normalization_type='GroupNormalization', weight_standardization=True):
    block_to_upsample = [x]
    for l in range(number_of_layers):
        la = get_tiramisu_layer(x, filters, padding=padding, activation=activation, convolution_type=convolution_type, dropout_rate=dropout_rate, use_bias=use_bias, kernel_initializer=kernel_initializer, weight_decay=weight_decay, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, normalization_type=normalization_type, weight_standardization=weight_standardization)
        block_to_upsample.append(la)
        x = layers.Concatenate(axis=3)([x, la])
    return x, block_to_upsample
        
def get_transition_down(x, filters, filter_size=(1, 1), padding='same', activation='relu', convolution_type='Conv2D', dropout_rate=0.2, use_bias=False, kernel_initializer='he_normal', kernel_regularizer='l2', weight_decay=1e-4, bn_momentum=0.9, bn_epsilon=1.1e-5, pool_size=2, strides=2, normalization_type='GroupNormalization', weight_standardization=True):
    if filter_size==(1, 1) or filter_size==1:
        convolution_type = 'Conv2D'
    x = get_tiramisu_layer(x, filters, filter_size=filter_size, padding=padding, activation=activation, convolution_type=convolution_type, dropout_rate=dropout_rate, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, weight_decay=weight_decay, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, normalization_type=normalization_type, weight_standardization=weight_standardization)
    x = layers.MaxPooling2D(pool_size=pool_size, strides=strides, padding=padding)(x)
    return x
    
def get_transition_up(skip_connection, block_to_upsample, filters, padding="same", activation='relu', kernel_initializer='he_normal', kernel_regularizer='l2', weight_decay=1e-4, **kwargs):
    x = layers.Concatenate(axis=3)(block_to_upsample[1:])
    x = layers.Conv2DTranspose(filters, kernel_size=3, strides=2, padding=padding, activation=activation, kernel_regularizer=get_kernel_regularizer(kernel_regularizer, weight_decay))(x)
    x = layers.Concatenate(axis=3)([x, skip_connection])
    return x 

def get_normalization_layer(x, normalization_type, bn_momentum=0.9, bn_epsilon=1.1e-5, gn_groups=16):
    if normalization_type in ['BN', 'BatchNormalization']:
        x = layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(x)
    elif normalization_type in ['GN', 'GroupNormalization']:
        x = tfa.layers.GroupNormalization(groups=find_number_of_groups(x.shape[-1], gn_groups))(x)
    elif normalization_type == 'BCN':
        x = tfa.layers.GroupNormalization(groups=find_number_of_groups(x.shape[-1], gn_groups))(x)
        x = layers.BatchNormalization(momentum=bn_momentum, epsilon=bn_epsilon)(x)
    return x

def get_uncompiled_tiramisu(nfilters=48, growth_rate=16, layers_scheme=[4, 5, 7, 10, 12], bottleneck=15, activation='relu', convolution_type='SeparableConv2D', padding='same', last_convolution=False, dropout_rate=0.2, weight_standardization=True, model_img_size=(None, None), input_channels=3, use_bias=False, kernel_initializer='he_normal', kernel_regularizer='l2', weight_decay=1e-4, heads=[{'name': 'crystal', 'type': 'segmentation'}, {'name': 'loop_inside', 'type': 'segmentation'}, {'name': 'loop', 'type': 'segmentation'}, {'name': 'stem', 'type': 'segmentation'}, {'name': 'pin', 'type': 'segmentation'}, {'name': 'foreground', 'type': 'segmentation'}], verbose=False, name='model', normalization_type='GroupNormalization', gn_groups=16, bn_momentum=0.9, bn_epsilon=1.1e-5, input_dropout=0.):
    print('get_uncompiled_tiramisu heads', heads)
    boilerplate={'activation': activation, 'convolution_type': convolution_type, 'padding': padding, 'dropout_rate': dropout_rate, 'use_bias': use_bias, 'kernel_initializer': kernel_initializer, 'kernel_regularizer': kernel_regularizer, 'weight_decay': weight_decay, 'normalization_type': normalization_type, 'weight_standardization': weight_standardization}
    
    inputs = keras.Input(shape=(model_img_size) + (input_channels,))
    
    nfilters_start = nfilters

    if input_dropout > 0.:
        x = layers.Dropout(dropout_rate=input_dropout)(inputs)
    else:
        x = inputs
        
    x = get_tiramisu_layer(x, nfilters, filter_size=3, padding=padding, activation=activation, convolution_type='Conv2D', dropout_rate=dropout_rate, use_bias=use_bias, kernel_initializer=kernel_initializer, weight_decay=weight_decay, bn_momentum=bn_momentum, bn_epsilon=bn_epsilon, normalization_type=normalization_type, weight_standardization=weight_standardization)
    
    _skips = []
    for l, number_of_layers in enumerate(layers_scheme):
        x, block_to_upsample = get_dense_block(x, growth_rate, number_of_layers, **boilerplate)
        _skips.append(x)
        nfilters += number_of_layers*growth_rate
        x = get_transition_down(x, nfilters, **boilerplate)
        if verbose:
            print('layer:', l, number_of_layers, 'shape:', x.shape)
    x, block_to_upsample = get_dense_block(x, growth_rate, bottleneck, **boilerplate)
    if verbose:
        print('bottleneck:', l, number_of_layers, 'shape:', x.shape)
    _skips = _skips[::-1]
    extended_layers_scheme = layers_scheme + [bottleneck]
    extended_layers_scheme.reverse()
    for l, number_of_layers in enumerate(layers_scheme[::-1]):
        n_filters_keep = growth_rate * extended_layers_scheme[l]
        if verbose:
            print('n_filters_keep', n_filters_keep)
        x = get_transition_up(_skips[l], block_to_upsample, n_filters_keep)
        x_up, block_to_upsample = get_dense_block(x, growth_rate, number_of_layers, **boilerplate)
        if verbose:
            print('layer:', l, number_of_layers, 'shape:', x.shape, 'x_up.shape', x_up.shape)
    
    outputs = []
    regression_neck = None
    for head in heads:
        if head['type'] == 'segmentation' or head['type'] == 'click_segmentation':
            
            output = layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype="float32", name=head['name'])(x_up)
            
            #output = get_convolutional_layer(x_up, 'Conv2D', 1, filter_size=1, padding=padding, use_bias=use_bias, kernel_initializer=kernel_initializer, kernel_regularizer=kernel_regularizer, weight_decay=weight_decay, weight_standardization=weight_standardization, activation="sigmoid", dtype="float32", name=head['name'])
            
        elif head['type'] == 'regression':
            if regression_neck is None:
                #regression_neck = layers.Conv2D(1, 1, activation="sigmoid", padding="same", dtype="float32", name=head['name'])(x_up)
                regression_neck = get_tiramisu_layer(x_up, 1, 1, activation="sigmoid", convolution_type='Conv2D')
                batch_size, input_shape_y, input_shape_w, channels = K.shape(regression_neck).numpy
                target_shape = (batch_size, 224, 224, channels)
                target_placeholder = K.placeholder(shape=target_shape)
                regression_neck = UpsampleLike(name='resize_regression')([regression_neck, target_placeholder])
                regression_neck = layers.Flatten()(regression_neck)
                regression_neck = layers.Dropout(dropout_rate)(regression_neck)
                #regression_neck = get_transition_down(regression_neck, 512, strides=11, pool_size=11)
                #regression_neck = layers.GlobalMaxPool2D()(regression_neck)
                #regression_neck = layers.Flatten()(regression_neck)
            output = layers.Dense(3, activation="sigmoid", dtype="float32", name=head['name'])(regression_neck)
        outputs.append(output)
    model = keras.Model(inputs=inputs, outputs=outputs, name=name)
    return model

def get_tiramisu(nfilters=48, growth_rate=16, layers_scheme=[4, 5, 7, 10, 12], bottleneck=15, activation='relu', convolution_type='Conv2D', last_convolution=False, dropout_rate=0.2, weight_standardization=True, model_img_size=(None, None), use_bias=False, learning_rate=0.001, finetune=False, finetune_model=None, heads=[{'name': 'crystal', 'type': 'segmentation'}, {'name': 'loop_inside', 'type': 'segmentation'}, {'name': 'loop', 'type': 'segmentation'}, {'name': 'stem', 'type': 'segmentation'}, {'name': 'pin', 'type': 'segmentation'}, {'name': 'capillary', 'type': 'segmentation'}, {'name': 'ice', 'type': 'segmentation'}, {'name': 'foreground', 'type': 'segmentation'}, {'name': 'click', 'type': 'click_segmentation'}], name='model', normalization_type='GroupNormalization', limit_loss=True, weight_decay=1.e-4):
    print('get_tiramisu heads', heads)
    model = get_uncompiled_tiramisu(nfilters=nfilters, growth_rate=growth_rate, layers_scheme=layers_scheme, bottleneck=bottleneck, activation=activation, convolution_type=convolution_type, last_convolution=last_convolution, dropout_rate=dropout_rate, weight_standardization=weight_standardization, model_img_size=model_img_size, heads=heads, name=name, normalization_type=normalization_type, weight_decay=weight_decay)
    if finetune and finetune_model is not None:
        print('loading weights to finetune')
        model.load_weights(finetune_model)
    else:
        print('not finetune')
    losses = {}
    metrics = {}
    for head in heads:
        losses[head['name']] = params[head['type']]['loss']
        if params[head['type']]['metrics'] == 'BIoU':
            metrics[head['name']] = [tf.keras.metrics.BinaryIoU(target_class_ids=[1], threshold=0.5, name='BIoU_1'),
                                     tf.keras.metrics.BinaryIoU(target_class_ids=[0], threshold=0.5, name='BIoU_0'),
                                     tf.keras.metrics.BinaryIoU(target_class_ids=[0, 1], threshold=0.5, name='BIoU_both')]
        elif params[head['type']]['metrics'] == "BIoUm": 
            metrics[head['name']] = [tf.keras.metrics.BinaryIoUm(target_class_ids=[1], threshold=0.5, name='BIoUm_1'),
                                     tf.keras.metrics.BinaryIoUm(target_class_ids=[0], threshold=0.5, name='BIoUm_0'),
                                     tf.keras.metrics.BinaryIoUm(target_class_ids=[0, 1], threshold=0.5, name='BIoUm_both')]
        elif params[head['type']]['metrics'] == 'mean_absolute_error':
            metrics[head['name']] = [tf.keras.metrics.mean_absolute_error()]
                                                
    print('losses', len(losses), losses)
    print('metrics', len(metrics), metrics)
    loss_weights = {}
    for head in heads:
        lw = loss_weights_from_stats[head['name']]
        if limit_loss:
            if lw > loss_weights_from_stats['crystal']:
                lw = loss_weights_from_stats['crystal']
        loss_weights[head['name']] = lw
        
    lrs = learning_rate
    #lrs = tf.keras.optimizers.schedules.ExponentialDecay(lrs, decay_steps=1e4, decay_rate=0.96, minimum_value=1e-7, staircase=True)
    optimizer = tf.keras.optimizers.RMSprop(learning_rate=lrs)
    #optimizer = tf.keras.optimizers.Adam(learning_rate=lrs)
    if finetune:
        for l in model.layers[:-len(heads)]:
            l.trainable = False
            
    model.compile(optimizer=optimizer, loss=losses, loss_weights=loss_weights, metrics=metrics) 
    
    print('model.losses', len(model.losses), model.losses)
    print('model.metrics', len(model.metrics), model.metrics)
    return model

def path_to_input_image(path):
    return img_to_array(load_img(path, target_size=img_size))

def path_to_target(path):
    img = img_to_array(
        load_img(path, target_size=img_size, color_mode="grayscale"))
    img = img.astype("uint8")
    return img

def get_paths(directory='images_and_labels', seed=1337):
    input_img_paths = glob.glob(os.path.join(directory, '*/img.jpg'))
    target_img_paths = [item.replace('img.jpg', 'foreground.png') for item in input_img_paths]
    random.Random(seed).shuffle(input_img_paths)
    random.Random(seed).shuffle(target_img_paths)
    return input_img_paths, target_img_paths

def get_training_dataset(seed=1337, num_val_samples=150):
    input_img_paths, target_img_paths = get_paths(seed=seed)
    train_paths = input_img_paths[:-num_val_samples]
    train_target_img_paths = target_img_paths[:-num_val_samples]
    return train_paths, train_target_img_paths

def get_validation_dataset(seed=1337, num_val_samples=150):
    input_img_paths, target_img_paths = get_paths(seed=seed)
    val_paths = input_img_paths[-num_val_samples:]
    val_target_img_paths = target_img_paths[-num_val_samples:]
    return val_paths, val_target_img_paths
    
def get_family(name):
    fname = os.path.realpath(name)
    search_string = '.*/double_clicks_(.*)_double_click.*|.*/(.*)_manual_omega.*|.*/(.*)_color_zoom.*|.*/(.*)_auto_omega.*'
    match = re.findall(search_string, fname)
    print('match', match)
    if match:
        for item in match[0]:
            if item != "":
                return item
    else:
        return os.path.basename(os.path.dirname(fname))
    
def get_sample_families(directory='images_and_labels', subset_designation="*"):
    search_string = '{directory:s}/double_clicks_(.*)_double_click.*|{directory:s}/(.*)_manual_omega.*|{directory:s}/(.*)_color_zoom.*|{directory:s}/(.*)_auto_omega.*'.format(directory=directory)
    individuals = glob.glob('%s/%s' % (directory, subset_designation))
    sample_families = {}
    for individual in individuals:
         matches = re.findall(search_string, individual)
         individual = individual.replace('%s/' % directory, '')
         if matches:
             for match in matches[0]:
                 if match != '':
                     if match in sample_families:
                         sample_families[match].append(individual)
                     else:
                         sample_families[match] = [individual]
         else:
             sample_families[individual] = [individual]
    return sample_families
    
def get_paths_for_families(families_subset_list, sample_families, directory):
    paths = []
    for family in families_subset_list:
        for individual in sample_families[family]:
            paths.append(os.path.join(directory, individual, 'img.jpg'))
    return paths

def get_training_and_validation_datasets(directory='images_and_labels', seed=12345, split=0.2):
    sample_families = get_sample_families(directory=directory)
    sample_families_names = list(sample_families.keys())
    sample_families_names.sort()
    random.Random(seed).shuffle(sample_families_names)
    total = len(sample_families_names)
    
    train = int((1-split)*total)
    train_families = sample_families_names[:train]
    valid_families = sample_families_names[train:]
    print('total %d' % total)
    print('train', train)
    print('train_families: %d' % len(train_families))
    print('valid_families: %d' % len(valid_families))
    
    train_paths = get_paths_for_families(train_families, sample_families, directory)
    random.Random(seed).shuffle(train_paths)
    val_paths = get_paths_for_families(valid_families, sample_families, directory)
    random.Random(seed).shuffle(val_paths)
    
    return train_paths, val_paths
    
def get_training_and_validation_datasets_for_clicks(basedir='./', seed=1, background_percent=10, train_images=10000, valid_images=2500, forbidden=[]):
    
    backgrounds = glob.glob(os.path.join(basedir, 'shapes_of_background/*.jpg')) + glob.glob(os.path.join(basedir, 'Backgrounds/*.jpg'))
    random.Random(seed).shuffle(backgrounds)
    train_paths = glob.glob(os.path.join(basedir, 'unique_shapes_of_clicks/*.jpg')) #+ glob.glob('images_and_labels_augmented/*/img.jpg')
    random.Random(seed).shuffle(train_paths)
    train_paths = train_paths[:train_images]
    backgrounds = backgrounds[:int(len(train_paths)/background_percent)]
    train_paths += backgrounds
    random.Random(seed).shuffle(train_paths)
    val_paths = train_paths[-valid_images:]
    if len(train_paths) - valid_images < train_images:
        train_paths = train_paths[:train_images]
    else:
        train_paths = train_paths[:-valid_images]
    if len(forbidden) > 0:
        train_paths = [item for item in train_paths if item not in forbidden]
    return train_paths, val_paths

def segment_multihead(base='/nfs/data2/Martin/Research/murko', epochs=25, patience=3, mixed_precision=False, name='start', source_weights=None, batch_size=16, model_img_size=(512, 512), network='fcdn56', convolution_type='SeparableConv2D',  heads=[{'name': 'crystal', 'type': 'segmentation'}, {'name': 'loop_inside', 'type': 'segmentation'}, {'name': 'loop', 'type': 'segmentation'}, {'name': 'stem', 'type': 'segmentation'}, {'name': 'pin', 'type': 'segmentation'}, {'name': 'capillary', 'type': 'segmentation'}, {'name': 'ice', 'type': 'segmentation'}, {'name': 'foreground', 'type': 'segmentation'}, {'name': 'click', 'type': 'click_segmentation'}], notions=['crystal', 'loop_inside', 'loop', 'stem', 'pin', 'capillary', 'ice', 'foreground', 'click'], last_convolution=False, augment=False, train_images=-1, valid_images=1000, scale_click=False, click_radius=320e-3, learning_rate=0.001, pixel_budget=768*992, normalization_type='GroupNormalization', validation_scale=0.4, dynamic_batch_size=True, finetune=False, seed=12345, artificial_size_increase=1, include_plate_images=False, include_capillary_images=False, dropout_rate=0.2, weight_standardization=True, limit_loss=True, weight_decay=1.e-4):
    
    if mixed_precision:
        print('setting mixed_precision')
        keras.mixed_precision.set_global_policy("mixed_float16")
    
    for gpu in tf.config.list_physical_devices('GPU'): 
        print('setting memory_growth on', gpu)
        tf.config.experimental.set_memory_growth(gpu, True)
        
    distinguished_name = "%s_%s" % (network, name)
    model_name = os.path.join(base, "%s.h5" % distinguished_name)
    history_name = os.path.join(base, "%s.history" % distinguished_name)
    png_name = os.path.join(base, "%s_losses.png" % distinguished_name)
    checkpoint_filepath = '%s_{batch:06d}_{loss:.4f}.h5' % distinguished_name
    #segment_train_paths, segment_val_paths = get_training_and_validation_datasets()
    #print('training on %d samples, validating on %d samples' % ( len(train_paths), len(val_paths)))
    # data genrators
    train_paths, val_paths = get_training_and_validation_datasets(directory='images_and_labels', split=0.2)
    if include_plate_images:
        train_paths_plate, val_paths_plate = get_training_and_validation_datasets(directory='images_and_labels_plate', split=0)
        #val_paths += val_paths_plate
        train_paths += train_paths_plate
    if include_capillary_images:
        train_paths_capillary, val_paths_capillary = get_training_and_validation_datasets(directory='images_and_labels_capillary', split=0.2)
        #val_paths += val_paths_plate
        train_paths += train_paths_capillary
        val_paths += val_paths_capillary
    full_size = len(train_paths)
    if train_images != -1:
        train_paths = train_paths[:train_images]
        factor = full_size//len(train_paths)
        train_paths = train_paths * (factor + 1)
        random.Random(seed).shuffle(train_paths)
        train_paths = train_paths[:full_size]
        
    #train_paths, val_paths = get_training_and_validation_datasets_for_clicks(basedir='/dev/shm', train_images=train_images, valid_images=valid_images, forbidden=[])
    print('\ntotal number of samples %d' % len(train_paths + val_paths))
    print('training on %d samples, validating on %d samples\n' % ( len(train_paths), len(val_paths)))
    #train_gen = CrystalClickDataset(batch_size, model_img_size, train_paths, augment=augment, scale_click=scale_click, click_radius=click_radius, dynamic_batch_size=dynamic_batch_size, shuffle_at_0=True)
    notions = [item['name'] for item in heads]
    print('notions in segment_multihead', notions)
    train_gen = MultiTargetDataset(batch_size, model_img_size, train_paths, notions=notions, augment=augment, scale_click=scale_click, click_radius=click_radius, dynamic_batch_size=dynamic_batch_size, pixel_budget=pixel_budget, artificial_size_increase=artificial_size_increase, shuffle_at_0=True)
    val_model_img_size = get_img_size_as_scale_of_pixel_budget(validation_scale)
    val_batch_size = get_dynamic_batch_size(val_model_img_size)
    #val_gen = CrystalClickDataset(val_batch_size, val_model_img_size, val_paths, augment=False, scale_click=scale_click, click_radius=click_radius, dynamic_batch_size=False)
    val_gen = MultiTargetDataset(val_batch_size, val_model_img_size, val_paths, augment=False, notions=notions, pixel_budget=pixel_budget)
    # callbacks
    checkpointer = keras.callbacks.ModelCheckpoint(model_name, verbose=1, monitor='val_loss', save_best_only=True, mode='min')
    #checkpointer2 = keras.callbacks.ModelCheckpoint(filepath=checkpoint_filepath, verbose=1, monitor='loss', save_freq=2000, save_best_only=False, mode='min')
    nanterminator = keras.callbacks.TerminateOnNaN()
    #tensorboard = keras.callbacks.TensorBoard(log_dir=os.path.join(os.path.realpath('./'), '%s_logs' % model_name.replace('.h5', '')), update_freq='epoch', write_steps_per_second=True)
    #earlystopper = keras.callbacks.EarlyStopping(patience=patience, verbose=1)
    lrreducer = keras.callbacks.ReduceLROnPlateau(factor=0.75,monitor="val_loss", patience=patience, cooldown=1, min_lr=1e-6, verbose=1),
    callbacks = [checkpointer, nanterminator, lrreducer]
    network_parameters = networks[network]
    
    if os.path.isdir(model_name) or os.path.isfile(model_name):
        print('model exists, loading weights ...')
        #model = keras.models.load_model(model_name)
        model = get_tiramisu(convolution_type=convolution_type, model_img_size=(None, None), heads=heads, last_convolution=last_convolution, name=network, learning_rate=learning_rate, dropout_rate=dropout_rate, weight_standardization=weight_standardization,normalization_type=normalization_type, finetune=finetune, finetune_model=model_name, limit_loss=limit_loss, weight_decay=weight_decay, **network_parameters)
        if not finetune:
            model.load_weights(model_name)
        history_name = history_name.replace('.history', '_next_superepoch.history')
        png_name = png_name.replace('.png', '_next_superepoch.png')
    else:
        print(model_name, 'does not exist')
        #custom_objects = {"click_loss": click_loss, "ClickMetric": ClickMetric}
        #with keras.utils.custom_object_scope(custom_objects):
        model = get_tiramisu(convolution_type=convolution_type, model_img_size=(None, None), heads=heads, last_convolution=last_convolution, name=network, learning_rate=learning_rate, dropout_rate=dropout_rate, weight_standardization=weight_standardization, normalization_type=normalization_type, limit_loss=limit_loss, weight_decay=weight_decay, **network_parameters)
        #sys.exit()
    print(model.summary())
        
    history = model.fit(train_gen, 
                        epochs=epochs, 
                        validation_data=val_gen, 
                        callbacks=callbacks, 
                        max_queue_size=64, 
                        use_multiprocessing=True, 
                        workers=32)
    
    f = open(history_name, "wb")
    pickle.dump(history.history, f)
    f.close()
    
    epochs = range(1, len(history.history["loss"]) + 1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure(figsize=(16,9))
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig(png_name)
    plot_history(png_name.replace('_losses.png', ''), history.history)
    
def segment(base='/nfs/data2/Martin/Research/murko', epochs=25, patience=5):
    keras.mixed_precision.set_global_policy("mixed_float16")
    date = 'start'
    date_next = '2021-12-13_tiramisu_SeparableConv2D'
    model_and_weights = os.path.join(base, "sample_segmentation_%s.h5" % date)
    next_model_and_weights = os.path.join(base, "sample__segmentation_%s.h5" % date_next)
    model_train_history = os.path.join(base, "sample_segmentation_%s_history.pickle" % date)
    #model = get_other_model(model_img_size, num_classes)
    # Split our img paths into a training and a validation set
    #train_paths, train_target_img_paths = get_training_dataset()
    #val_paths, val_target_img_paths = get_validation_dataset()
    train_paths, val_paths = get_training_and_validation_datasets()
    print('training on %d samples, validating on %d samples' % ( len(train_paths), len(val_paths)))
    
    # Instantiate data Sequences for each split
    train_gen = SampleSegmentationDataset(batch_size, model_img_size, train_paths, augment=True)
    val_gen = SampleSegmentationDataset(batch_size, model_img_size, val_paths, augment=False)
    
    model = get_tiramisu(convolution_type='SeparableConv2D')
    

    #new_date = '2021-11-12_zoom123'
    checkpointer = keras.callbacks.ModelCheckpoint(next_model_and_weights, verbose=1, mode='min', save_best_only=True)
    earlystopper = keras.callbacks.EarlyStopping(patience=patience, verbose=1)
    callbacks = [checkpointer, earlystopper]

    # Train the model, doing validation at the end of each epoch.
    #strategy = tf.distribute.MirroredStrategy()
    #with strategy.scope():
        #model = get_tiramisu(convolution_type='SeparableConv2D')
        #if os.path.isfile(model_and_weights):
            #model.load_weights(model_and_weights)
        #history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    history = model.fit(train_gen, epochs=epochs, validation_data=val_gen, callbacks=callbacks)
    f = open(model_train_history, "wb")
    pickle.dump(history.history, f)
    f.close()
    # Generate predictions for all images in the validation set
    #val_gen = SampleSegmentationDataset(batch_size, img_size, val_paths, val_target_img_paths)
    _start = time.time()
    val_preds = model.predict(val_gen)
    print('predicting %d examples took %.4f seconds' % (len(val_preds), time.time() - _start))
    epochs = range(1, len(history.history["loss"]) + 1)
    loss = history.history["loss"]
    val_loss = history.history["val_loss"]
    plt.figure()
    plt.plot(epochs, loss, "bo-", label="Training loss")
    plt.plot(epochs, val_loss, "ro-", label="Validation loss")
    plt.title("Training and validation loss")
    plt.legend()
    plt.savefig('sample_foreground_segmentation_%s.png' % date)
    plt.show()

def efficient_resize(img, new_size, anti_aliasing=True):
    return img_to_array(array_to_img(img).resize(new_size[::-1]), dtype='float32')/255.

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
            present = -1
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

def get_most_likely_click(predictions, verbose=False):
    _start = time.time()
    gmlc = False
    most_likely_click = -1, -1
    shape=predictions[0].shape[1: 3]
    for notion in ['crystal', 'loop_inside', 'loop']:
        notion_prediction = get_notion_prediction(predictions, notion)
        if notion_prediction[0] == 1:
            most_likely_click = notion_prediction[1]/shape[0], notion_prediction[2]/shape[1]
            if verbose:
                print('%s found!' % notion)
            gmlc = True
            break
    if gmlc is False:
        foreground = get_notion_prediction(predictions, 'foreground')
        if foreground[0] == 1:
            most_likely_click = foreground[5]/shape[0], foreground[6]/shape[1]
    if verbose:
        print('most likely click determined in %.4f seconds' % (time.time() - _start))
    return most_likely_click

def get_loop_bbox(predictions):
    loop_present, r, c, h, w, r_max, c_max, area, notion_prediction = get_notion_prediction(predictions, 'loop')
    shape = predictions[0].shape[1:3]
    r /= shape[0]
    c /= shape[1]
    h /= shape[0]
    w /= shape[1]
    return loop_present, r, c, h, w

def predict_multihead(to_predict=None, image_paths=None, base='/nfs/data2/Martin/Research/murko', model_name='fcdn103_256x320_loss_weights.h5', directory='images_and_labels', nimages=-1, batch_size=16, model_img_size=(224, 224), augment=False, threshold=0.5, train=False, split=0.2, target=False, model=None, save=True, prefix='prefix'):
    
    _start = time.time()
    if model is None:
        model = keras.models.load_model(os.path.join(base, model_name), custom_objects={'WSConv2D': WSConv2D, 'WSSeparableConv2D': WSSeparableConv2D})
        print('model loaded in %.4f seconds' % (time.time() - _start))
    
    notions = [layer.name for layer in model.layers[-10:] if isinstance(layer, keras.layers.Conv2D)]
    notion_indices = dict([(notion, notions.index(notion)) for notion in notions])
    notion_indices['click'] = -1
    
    model_img_size = get_closest_working_img_size(model_img_size)
    print('model_img_size will be', model_img_size)
    
    all_image_paths = []
    if to_predict is None:
        train_paths, val_paths = get_training_and_validation_datasets(directory, split=split)
        if train:
            to_predict = train_paths
        else:
            to_predict = val_paths
    elif not type(to_predict) is list and not type(to_predict) is np.ndarray and os.path.isdir(to_predict):
        to_predict = glob.glob(os.path.join(to_predict, '*.jpg'))
    elif not type(to_predict) is list and not type(to_predict) is np.ndarray and os.path.isfile(to_predict):
        all_image_paths.append(os.path.realpath(to_predict))
        to_predict = np.expand_dims(get_img(to_predict, size=model_img_size), 0)
    elif type(to_predict) is bytes and simplejpeg.is_jpeg(to_predict):
        img_array = simplejpeg.decode_jpeg(to_predict)
        to_predict = np.expand_dims(img_array, 0)
    elif type(to_predict) is list:
        if simplejpeg.is_jpeg(to_predict[0]):
            to_predict = [simplejpeg.decode_jpeg(jpeg) for jpeg in to_predict]
        elif os.path.isfile(to_predict[0]):
            to_predict = [img_to_array(load_img(img, target_size=model_img_size), dtype='float32') for img in to_predict]
        if type(to_predict[0]) is np.ndarray:
            to_predict = [img.astype("float32")/255. for img in to_predict]
        if size_differs(to_predict[0].shape[:2], model_img_size):
            to_predict = [efficient_resize(img, model_img_size, anti_aliasing=True) for img in to_predict]
        to_predict = np.array(to_predict)
    elif type(to_predict) is np.ndarray:
        if len(to_predict.shape) == 3:
            to_predict = np.expand_dims(to_predict, 0)
        if size_differs(to_predict[0].shape[:2], model_img_size):
            to_predict = np.array([efficient_resize(img, model_img_size, anti_aliasing=True) for img in to_predict])
    print('all_images ready for prediction in %.4f seconds' % (time.time()-_start))
    
    all_input_images = []
    all_ground_truths = []
    all_predictions = []
    
    _start_predict = time.time()
    if type(to_predict) is np.ndarray:
         all_predictions = model.predict(to_predict)
         
    elif type(to_predict) is list:
        if batch_size == -1:
            batch_size = get_dynamic_batch_size(model_img_size)
        gen = MultiTargetDataset(batch_size, model_img_size, to_predict, notions=notions, augment=augment, target=target)
        for i, (input_images, ground_truths) in enumerate(gen):
            _start = time.time()
            predictions = model.predict(input_images)
            #, use_multiprocessing=True, workers=batch_size, max_queue_size=2*batch_size)
            all_input_images = np.vstack([all_input_images, input_images]) if all_input_images else input_images
            all_ground_truths = np.vstack([all_ground_truths, ground_truths]) if all_ground_truths else ground_truths
            all_predictions = np.vstack([all_predictions, predictions]) if all_predictions else predictions
            all_image_paths += gen.batch_img_paths
    
    end = time.time()
    print('%d images predicted in %.4f seconds (%.4f per image)' % (len(to_predict), end-_start_predict, (end-_start_predict)/len(to_predict)))
    
    if save:
        if not all_image_paths:
            all_image_paths = ['/tmp/%d_%s.jpg' % (k, prefix) for k in range(len(to_predict))]
        if not all_input_images:
            all_input_images = to_predict
        save_predictions(all_input_images, all_predictions, all_image_paths, all_ground_truths, notions, notion_indices, model_img_size, train=train, target=target)
    
    return all_predictions 

def get_hierarchical_mask_from_target(target, notions=['crystal', 'loop_inside', 'loop', 'stem', 'pin', 'capillary', 'ice', 'foreground', 'click'], notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'capillary': 5, 'ice': 6, 'foreground': 7, 'click':-1}):
    hierarchical_mask = np.zeros(target.shape[:2], dtype=np.uint8)
    for notion in notions[:-1][::-1]:
        l = notion_indices[notion]
        mask = target[:,:,l]
        if np.any(mask):
            hierarchical_mask[mask==1] = notions[:-1][::-1].index(notion) + 1
        hierarchical_mask[0, notions.index(notion)] = notions.index(notion) + 1
    return hierarchical_mask

def get_kth_prediction_from_predictions(k, predictions):
    prediction = np.zeros(predictions.shape[1:3] + predictions.shape[0], dtype=np.uint8)
    for n, notion in enumerate(predictions):
        prediction[:,:,n] = predictions[n][k][:,:,0]
    return prediction
        
def massage_mask(mask, min_size=32, massager='convex'):
    mask = remove_small_objects(mask, min_size=min_size)
    if not np.any(mask):
        return mask
    labeled_image = mask.astype('uint8')
    properties = regionprops(labeled_image)[0]
    bbox = properties.bbox
    if massager == 'convex':
        mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = properties.convex_image
    elif massager == 'filled':
        mask[bbox[0]: bbox[2], bbox[1]: bbox[3]] = properties.filled_image
    return mask

def get_hierarchical_mask_from_predictions(predictions, k=0, notions=['crystal', 'loop_inside', 'loop', 'stem', 'pin', 'foreground'], notion_indices={'crystal': 0, 'loop_inside': 1, 'loop': 2, 'stem': 3, 'pin': 4, 'foreground': 5}, threshold=0.5, notion_values={'crystal': 7, 'loop_inside': 6, 'loop': 5, 'stem': 4, 'pin': 3, 'foreground': 8}, min_size=32, massage=True):
    hierarchical_mask = np.zeros(predictions[0].shape[1:3], dtype=np.uint8)
    for notion in notions[::-1]:
        notion_value = notion_values[notion]**2
        l = notion_indices[notion]
        mask = predictions[l][k,:,:,0]>threshold
        if massage:
            if notion in ['crystal', 'loop', 'loop_inside', 'stem', 'pin']:
                massager = 'convex'
            else:
                massager = 'filled'
            mask = massage_mask(mask, min_size=min_size, massager=massager)
        if np.any(mask):
            hierarchical_mask[mask==1] = notion_value
    return hierarchical_mask
    
def save_predictions(input_images, predictions, image_paths, ground_truths, notions, notion_indices, model_img_size, model_name='default', train=False, target=False, threshold=0.5, click_threshold=0.95):
    _start = time.time()
    for k, input_image in enumerate(input_images):
        hierarchical_mask = np.zeros(model_img_size, dtype=np.uint8)
        predicted_masks = np.zeros(model_img_size + (len(notions),), dtype=np.uint8)
        if 'click' in notions:
            notions_in_order = notions[:-1][::-1] + [notions[-1]]
        else:
            notions_in_order = notions[::-1]
        for notion in notions_in_order :
            notion_value = notions.index(notion) + 1
            l = notion_indices[notion]
            if l != -1:
                mask = (predictions[l][k]>threshold)[:,:,0]
                predicted_masks[:,:,l] = mask
            else:
                mask = (predictions[l][k]>click_threshold)[:,:,0]
                predicted_masks[:,:,l] = mask
            if np.any(mask):
                hierarchical_mask[mask==1] = notion_value
            hierarchical_mask[-1, -(1+notions.index(notion))] = notion_value
        if target:
            label_mask = np.zeros(model_img_size, dtype=np.uint8)
            for notion in notions_in_order:
                notion_value = notions.index(notion) + 1
                l = notion_indices[notion]
                if l != -1:
                    mask = (ground_truths[l][k]>threshold)[:,:,0]
                else:
                    mask = (predictions[l][k]>click_threshold)[:,:,0]
                if np.any(mask):
                    label_mask[mask==1] = notion_value
                label_mask[-1, -(1+notions.index(notion))] = notion_value
        
        name = os.path.basename(image_paths[k])
        prefix = name[:-4]
        directory = os.path.dirname(image_paths[k])
        
        if train:
            prefix += '_train'
        
        template = '%s_%s_model_img_size_%dx%d' % (prefix, model_name.replace('.h5', ''), model_img_size[0], model_img_size[1])
        
        prediction_img_path = os.path.join(directory, '%s_hierarchical_mask_high_contrast_predicted.png' % (template))
        save_img(prediction_img_path, np.expand_dims(hierarchical_mask, axis=2), scale=True)
        
        predicted_masks_name = os.path.join(directory, '%s.npy' % template)
        np.save(predicted_masks_name, predicted_masks)
        
        if target:
            fig, axes = plt.subplots(1, 3)
        else:
            fig, axes = plt.subplots(1, 2)
        fig.set_size_inches(16, 9)
        title = name
        fig.suptitle(title)
        axes[0].set_title('input image with predicted click and loop bounding box (if any)')
        axes[0].imshow(input_image)
        axes[1].set_title('raw segmentation result with most likely click (if any)')
        axes[1].imshow(hierarchical_mask)
        if target:
            axes[2].set_title('ground truth')
            axes[2].imshow(label_mask)
            
        for a in axes.flatten():
            a.axis('off')
            
        original_shape = np.array(input_image.shape[:2])
        prediction_shape = np.array(hierarchical_mask.shape)
        most_likely_click = np.array(get_most_likely_click(predictions))
        if -1 not in most_likely_click:
            mlc_ii = most_likely_click*original_shape
            click_patch_ii = plt.Circle(mlc_ii[::-1], radius=2, color='red')
            axes[0].add_patch(click_patch_ii)
            
            mlc_hm = most_likely_click*prediction_shape
            click_patch_hm = plt.Circle(mlc_hm[::-1], radius=2, color='green')
            axes[1].add_patch(click_patch_hm)

        loop_present, r, c, h, w = get_loop_bbox(predictions)
        if loop_present != -1:
            r *= original_shape[0]
            c *= original_shape[1]
            h *= original_shape[0]
            w *= original_shape[1]
            loop_bbox_patch = plt.Rectangle((c-w/2, r-h/2), w, h, linewidth=1, edgecolor='green', facecolor='none')
            axes[0].add_patch(loop_bbox_patch)
            
        comparison_path = prediction_img_path.replace('hierarchical_mask_high_contrast_predicted', 'comparison')
        plt.savefig(comparison_path)
        plt.close()
    end = time.time()
    print('%d predictions saved in %.4f seconds (%.4f per image)' % (len(input_images), end-_start, (end-_start)/len(input_images)))
  
def get_img_size_as_scale_of_pixel_budget(scale, pixel_budget=768*992, ratio=0.75, modulo=32):
    n = math.floor(math.sqrt(pixel_budget/ratio))
    new_n = n*scale
    img_size = np.array((new_n*ratio, new_n)).astype(int)
    img_size -= np.mod(img_size, modulo)
    return tuple(img_size)

def get_img_size(resize_factor, original_size=(1024, 1360), modulo=32):
    new_size = resize_factor * np.array(original_size)
    new_size = get_closest_working_img_size(new_size, modulo=modulo)
    return new_size

def get_closest_working_img_size(img_size, modulo=32):
    closest_working_img_size = img_size - np.mod(img_size, modulo)
    return tuple(closest_working_img_size.astype(int))

def get_dynamic_batch_size(img_size, pixel_budget=768*992):
    return max(int(pixel_budget/np.prod(img_size)), 1)
    
def get_title_from_img_path(img_path):
    return os.path.basename(os.path.dirname(img_path))

def plot(sample, title='', k=0, notions=['crystal', 'loop_inside', 'loop', 'stem', 'pin', 'capillary', 'ice', 'foreground', 'click']):
     fig, axes = pylab.subplots(2, 5)
     fig.set_size_inches(24, 16)
     fig.suptitle(title)
     ax = axes.flatten()
     for a in ax:
         a.axis('off')
     ax[0].imshow(sample[0][k])
     ax[0].set_title('input image')
     for l in range(len(sample[1])):
         ax[1+l].imshow(sample[1][l][k][:,:,0])
         ax[1+l].set_title(notions[l])
     pylab.show()

def plot_augment(img_path, ntransformations=14, figsize=(24, 16), zoom_factor=0.5, shift_factor=0.5, shear_factor=45, rotate_probability=1, shear_probability=1, zoom_probability=1, shift_probability=1):
    fig, axes = pylab.subplots((2*ntransformations)//6+1, 6)
    fig.set_size_inches(*figsize)
    title = get_title_from_img_path(img_path)
    fig.suptitle(title)
    ax = axes.flatten()
    for a in ax:
        a.axis('off')
    img, target = get_img_and_target(img_path)
    ax[0].imshow(img)
    ax[0].set_title('input image')
    ax[1].imshow(get_hierarchical_mask_from_target(target))
    ax[1].set_title('original_target')
    
    for t in range(1, ntransformations+1):
        wimg = img[::]
        wtarget = target[::]
        if random.random()>0.5:
            wimg, wtarget = get_transposed_img_and_target(wimg, wtarget)
        if random.random()>0.5:
            wimg, wtarget = get_flipped_img_and_target(wimg, wtarget)
        wimg, wtarget = get_transformed_img_and_target(wimg, wtarget, shear_factor=shear_factor, zoom_factor=zoom_factor, shift_factor=shift_factor, rotate_probability=rotate_probability, shear_probability=shear_probability, zoom_probability=zoom_probability, shift_probability=shift_probability)
        
        ax[2*t].imshow(wimg)
        ax[2*t].set_title('%d input' % (t+1))
        ax[2*t+1].imshow(get_hierarchical_mask_from_target(wtarget))
        ax[2*t+1].set_title('%d target' % (t+1))
    
    pylab.show()
    

        
