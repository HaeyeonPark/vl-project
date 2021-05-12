#!/bin/bash
GPUS=0,1,2,3
export CUDA_VISIBLE_DEVICES=$GPUS

IMAGE_DIR=/workspace/data
BASE_ROOT=/workspace/code
ANNO_DIR=$BASE_ROOT/data/processed_data

CKPT_DIR=$BASE_ROOT/model_data/exp60
LOG_DIR=$BASE_ROOT/logs/exp60
PRETRAINED_PATH=$BASE_ROOT/pretrained/resnet50-19c8e357.pth
FOCAL_TYPE=none

lr=0.00012
num_epochs=60
batch_size=32
lr_decay_ratio=0.9
epochs_decay=20_30_40

num_classes=11003

python $BASE_ROOT/train.py \
    --CMPC \
    --CMPM \
    --pretrained \
    --model_path $PRETRAINED_PATH \
    --log_dir $LOG_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --checkpoint_dir $CKPT_DIR/lr-$lr-decay-$lr_decay_ratio-batch-$batch_size \
    --image_dir $IMAGE_DIR \
    --anno_dir $ANNO_DIR \
    --batch_size $batch_size \
    --gpus $GPUS \
    --num_epochs $num_epochs \
    --lr $lr \
    --lr_decay_ratio $lr_decay_ratio \
    --epochs_decay ${epochs_decay} \
    --num_classes ${num_classes} \
    --focal_type $FOCAL_TYPE \
    --feature_size 768 \
    --lambda_cont 0.1 \
    --lambda_combine 0.1 \
    --part2 2 \
    --part3 3 \
    ##--reranking
    ##--randsampling \
    ##--PART_CBT2I \
    ##--CONT \
    ##--COMBINE \
    ##--PART_I2T \



