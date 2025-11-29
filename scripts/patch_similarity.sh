#!/bin/bash

IMAGE_INDEX=21
PATCH_IDX=3904

python scripts/patch_similarity.py \
    --model_name vit_small \
    --checkpoint /home/admin/john/ijepa/logs/sen2venus_vits14.224-bs.128-ep.100_lr/jepa_sen2venus_lr-ep300.pth.tar \
    --dataset sen2venus \
    --image_index $IMAGE_INDEX \
    --patch_idx $PATCH_IDX \
    --dataset_root /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus \
    --save_path logs/patch_sim_${IMAGE_INDEX}_${PATCH_IDX}_lr_ep300.png \
    --alpha 0.6 \
    --no-use_hr \
    --split val \
    --img_size 896

#python scripts/patch_similarity.py \
#    --model_name vit_small \
#    --checkpoint /home/admin/john/ijepa/logs/sen2venus_vits14.224-bs.128-ep.100_hr/jepa_sen2venus-ep200.pth.tar \
#    --dataset sen2venus \
#    --image_index $IMAGE_INDEX \
#    --patch_idx $PATCH_IDX \
#    --dataset_root /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus \
#    --save_path logs/patch_sim_${IMAGE_INDEX}_${PATCH_IDX}_hr_ep200.png \
#    --alpha 0.6 \
#    --no-use_hr \
#    --split val \
#    --img_size 896

python scripts/patch_similarity.py \
    --model_name vit_small \
    --checkpoint /home/admin/john/ijepa/logs/sen2venus_vits14.224-bs.128-ep.100_hr_gram/jepa_sen2venus_lr_hr_gram-ep300.pth.tar \
    --dataset sen2venus \
    --image_index $IMAGE_INDEX \
    --patch_idx $PATCH_IDX \
    --dataset_root /home/admin/AGML_ResearchGroup/SuperResolution/sen2venus \
    --save_path logs/patch_sim_${IMAGE_INDEX}_${PATCH_IDX}_lrwgram_ep300.png \
    --alpha 0.6 \
    --no-use_hr \
    --split val \
    --img_size 896
