#!/bin/bash

# Paths and parameters
CHUNK_NAME="1_0"  # Replace with the actual chunk name
PROJECT_PATH="/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/raw"  # Replace with the base path to your project

# Chunks
CHUNK_PATH="${PROJECT_PATH}/camera_calibration/chunks/${CHUNK_NAME}"
BOUNDS_FILE="${CHUNK_PATH}"

# Input 
IMAGES_PATH="${PROJECT_PATH}/camera_calibration/rectified/images"
DEPTHS_PATH="${PROJECT_PATH}/camera_calibration/rectified/depths"
# ALPHA_MASKS_PATH="${PROJECT_PATH}/camera_calibration/rectified/masks"

# Output
OUTPUT_PATH="${PROJECT_PATH}/output"
HIERACHY_PATH="${OUTPUT_PATH}/chunks/${CHUNK_NAME}/hierarchy.hier"
SCAFFOLD_FILE="${OUTPUT_PATH}/scaffold/point_cloud/iteration_30000"
MODEL_PATH="${OUTPUT_PATH}/chunks/${CHUNK_NAME}_post"

# Parameters
SAVE_ITERATIONS="5_000"
CHECKPOINT_ITERATIONS="5_000"
ITERATIONS="5_000"

# Run the Python script
python -u train_post.py \
    -s "$CHUNK_PATH" \
    --model_path "$MODEL_PATH" \
    --hierarchy "$HIERACHY_PATH" \
    -i "$IMAGES_PATH" \
    -d "$DEPTHS_PATH" \
    --alpha_masks "$ALPHA_MASKS_PATH" \
    --scaffold_file "$SCAFFOLD_FILE" \
    --skybox_locked \
    --bounds_file "$BOUNDS_FILE" \
    --save_iterations $SAVE_ITERATIONS \
    --checkpoint_iterations $CHECKPOINT_ITERATIONS \
    --iterations $ITERATIONS
