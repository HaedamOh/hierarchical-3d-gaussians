#!/bin/bash

# Paths and parameters
# CHUNK_NAME="0_0"  # Replace with the actual chunk name
PROJECT_PATH="/home/shared/frontier_data/fnt_802/2024-12-13-10-24-rec001/processed/output_colmap"  
SOURCE_PATH="${PROJECT_PATH}/camera_calibration/aligned"

# Input
IMAGES_PATH="${PROJECT_PATH}/camera_calibration/rectified/images"
DEPTHS_PATH="${PROJECT_PATH}/camera_calibration/rectified/depths"
# ALPHA_MASKS_PATH="${PROJECT_PATH}/camera_calibration/rectified/masks"
SKYBOX_NUM=100000

# Output
OUTPUT_PATH="${PROJECT_PATH}/output"
MODEL_PATH="${OUTPUT_PATH}/scaffold" # Replace with the path to save the model

# Parameters
SAVE_ITERATIONS="30_000"
POSITION_LR_INIT=0.00016
POSITION_LR_FINAL=0.0000016

# Run the Python script
python -u train_coarse.py \
    -s "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --skybox_num "$SKYBOX_NUM" \
    --position_lr_init "$POSITION_LR_INIT" \
    --position_lr_final "$POSITION_LR_FINAL" \
    --images "$IMAGES_PATH" \
    --save_iterations $SAVE_ITERATIONS \
    --dataset_type "colmap" \


