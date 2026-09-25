#!/bin/bash
res=$1; 
batch=-1; 
fs=7; 
echo res ${res} batch ${batch} fs ${fs}

time ./train.py --name fixed_with_necks_i10_${res}x${res}_fs_${fs}\
   --model_img_size "(${res}, ${res})" \
   --val_model_img_size "(${res}, ${res})" \
   --filter_size ${fs} \
   --batch_size ${batch} \
   --resize_factor -1 \
   --learning_rate 0.0005 \
   -I 0.75 \
   -i 10 \
   --epochs 33 \
   --dataset \
       ../murko/manually_segmented_images/json/spine/soleil_proxima2a \
       ../murko/manually_segmented_images/json/spine/arthur_validated \
       ../murko/manually_segmented_images/json/spine/als_bl8.3.1 \
       ../murko/manually_segmented_images/json/spine/dls_i04 \
       ../murko/manually_segmented_images/json/spine/desy_p11 \
       ../murko/manually_segmented_images/json/spine/esrf_id30a \
       ../murko/manually_segmented_images/json/spine/bessy_bl14.1 \
       ../murko/manually_segmented_images/json/backgrounds \
   --train_dataset \
       ../murko/manually_segmented_images/json/spine/xrec \
       ../murko/manually_segmented_images/json/spine/c3d \
       ../murko/manually_segmented_images/json/spine/elettra_xrd2 \
       ../murko/manually_segmented_images/json/spine/sls \
       ../murko/manually_segmented_images/json/additional_backgrounds \
       ../murko/manually_segmented_images/json/spine/soleil_proxima1 \
   --not_multiprocessing \
   --workers 64 \
   --max_queue_size=256 \
   --dont_transform \
   --use_necks \

       #../murko/manually_segmented_images/json/plate \
       #../murko/manually_segmented_images/json/pcs/train_500 \

