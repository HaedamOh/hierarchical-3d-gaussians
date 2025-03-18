#!/bin/bash

# Paths and parameters
# CHUNK_NAME="0_0"  # Replace with the actual chunk name
PROJECT_PATH="./example_dataset"  # Replace with the base path to your project

SOURCE_PATH="${PROJECT_PATH}/camera_calibration/aligned"
BOUNDS_FILE="${SOURCE_PATH}"

# Input
IMAGES_PATH="${PROJECT_PATH}/camera_calibration/rectified/images"
DEPTHS_PATH="${PROJECT_PATH}/camera_calibration/rectified/depths"
# ALPHA_MASKS_PATH="${PROJECT_PATH}/camera_calibration/rectified/masks"

# Output
OUTPUT_PATH="${PROJECT_PATH}/output"
SCAFFOLD_FILE="${OUTPUT_PATH}/scaffold/point_cloud/iteration_30000"
MODEL_PATH="${OUTPUT_PATH}/aligned_entire" # Replace with the path to save the model

# Parameters
SAVE_ITERATIONS="15_000 30_000 50_000"
# CHECKPOINT_ITERATIONS="15_000 30_000 50_000" # .chkpt 

# START_CHECKPOINT="$MODEL_PATH/chkpnt10000.pth"

# Run the Python script
python -u train_single.py \
    -s "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --images "$IMAGES_PATH" \
    --depths "$DEPTHS_PATH" \
    --bounds_file "$BOUNDS_FILE" \
    --scaffold_file "$SCAFFOLD_FILE" \
    --alpha_masks "$ALPHA_MASKS_PATH" \
    --skybox_locked \
    --save_iterations $SAVE_ITERATIONS \
    --checkpoint_iterations $CHECKPOINT_ITERATIONS  \
    --dataset_type "colmap" \
    --iterations 50000 \
    # --start_checkpoint "$START_CHECKPOINT"

