#!/bin/bash

# Paths and parameters
PROJECT_PATH="/home/shared/frontier_data/fnt_802/2024-12-13-10-24-rec001/_raw"  # Replace with the base path to your project
SOURCE_PATH="${PROJECT_PATH}/camera_calibration/chunks/0_0"

# Input
IMAGES_PATH="${PROJECT_PATH}/camera_calibration/rectified/images"
DEPTHS_PATH="${PROJECT_PATH}/camera_calibration/rectified/depths"
# ALPHA_MASKS_PATH="${PROJECT_PATH}/camera_calibration/rectified/masks"


# Output
OUTPUT_PATH="${PROJECT_PATH}/output"

CHUNK_NAME="0_0" 
# HIERACHY_PATH="${OUTPUT_PATH}/chunks/merged_hierarchy.hier" # entire hierarchy 
HIERACHY_PATH="${OUTPUT_PATH}/chunks/${CHUNK_NAME}/hierarchy.hier_opt" # single chunk hierarchy
MODEL_PATH="${OUTPUT_PATH}/chunks/${CHUNK_NAME}" 

OUT_DIR="${OUTPUT_PATH}/chunks/${CHUNK_NAME}/render"  

# Parameters
SAVE_ITERATIONS="10_000 30_000"
CHECKPOINT_ITERATIONS="30_000" # .chkpt 

# START_CHECKPOINT="$MODEL_PATH/chkpnt10000.pth"

# Run the Python script
python -u render_hierarchy.py \
    -s "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --hierarchy "$HIERACHY_PATH" \
    -i "$IMAGES_PATH" \
    -d "$DEPTHS_PATH" \
    --dataset_type "colmap" \
    --out_dir "$OUT_DIR" \
    --eval 