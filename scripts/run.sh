#!/bin/bash

python train_net.py \
       --config-file ./configs/uavid/semantic-segmentation/elsa_ConvNeXtt_bs4_20k.yaml \
       --num-gpus 2 \
       DATALOADER.NUM_WORKERS 8
