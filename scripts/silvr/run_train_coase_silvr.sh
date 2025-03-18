#!/bin/bash

# Define project paths
PROJECT_PATH="/home/shared/nerfstudio/oxford_spires/2024-03-13-maths_2"
SOURCE_PATH="${PROJECT_PATH}"
IMAGES_PATH="${PROJECT_PATH}"

# Paths and parameters
SKYBOX_NUM=100000
TRANSFORMS_JSON="transforms_colmap_scaled_lidar_rect.json"

OUTPUT_PATH="${PROJECT_PATH}/output"
MODEL_PATH="${OUTPUT_PATH}/scaffold"  # Replace with the actual model path if necessary

# Iteration values for save and checkpoint
SAVE_ITERATIONS=30000  # Use numeric values
CHECKPOINT_ITERATIONS=30000  # Use numeric values

# Learning rate parameters
POSITION_LR_INIT=0.00016
POSITION_LR_FINAL=0.0000016

# Run the Python script
python train_coarse.py \
    -s "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --skybox_num "$SKYBOX_NUM" \
    --position_lr_init "$POSITION_LR_INIT" \
    --position_lr_final "$POSITION_LR_FINAL" \
    --images "$IMAGES_PATH" \
    --save_iterations "$SAVE_ITERATIONS" \
    --checkpoint_iterations "$CHECKPOINT_ITERATIONS" \
    --dataset_type "silvr" \
    --transforms_json "$TRANSFORMS_JSON"
