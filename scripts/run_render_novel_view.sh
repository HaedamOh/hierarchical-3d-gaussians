#!/bin/bash

# Paths and parameters
PROJECT_PATH="/home/shared/frontier_data/fnt_802/2024-12-13-10-24-rec001/processed/output_colmap"  # Replace with the base path to your project
SOURCE_PATH="${PROJECT_PATH}/camera_calibration/aligned_evo"

# Input
RENDER_PATH="/home/shared/frontier_data/fnt_802/multirecording_map/rendering/render_path" 
IMAGES_PATH="${PROJECT_PATH}/camera_calibration/rectified/images"
# DEPTHS_PATH="${PROJECT_PATH}/camera_calibration/rectified/depths"
# ALPHA_MASKS_PATH="${PROJECT_PATH}/camera_calibration/rectified/masks"


# Output
OUTPUT_PATH="${PROJECT_PATH}/output"
MODEL_PATH="${OUTPUT_PATH}/aligned_evo" # Replace with the path to save the model
OUT_DIR="${MODEL_PATH}/render"  

# Parameters
SAVE_ITERATIONS="10_000 30_000"
CHECKPOINT_ITERATIONS="30_000" # .chkpt 


# Run the Python script
python -u render_novel_view.py \
    -s "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    -i "$IMAGES_PATH" \
    --dataset_type "colmap" \
    --out_dir "$OUT_DIR" \
    --camera_novel_paths "$RENDER_PATH" \
    --eval 
