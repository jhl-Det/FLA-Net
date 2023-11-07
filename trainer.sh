#!/bin/bash
gpu_1='4'
gpu_2='9'

gpus=${gpu_1}

arch=UNet
export CUDA_VISIBLE_DEVICES=${gpus}

python train_w_hm.py \
--gpu_id ${gpus} \
--batchsize 2 \
--trainsize 352 \
--save_path ./cache/reproduce2/ \
--lr 2e-4 \
--arch ${arch} \
--epoch 200
