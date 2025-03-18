#!/bin/bash

# Paths and parameters
PROJECT_PATH="/home/shared/frontier_data/fnt_802/2024-12-13-10-24-rec001"
SOURCE_PATH="${PROJECT_PATH}/processed/output_colmap"  
IMAGES_PATH="${PROJECT_PATH}/raw/images"
DEPTHS_PATH="" # No depth atm
# ALPHA_MASKS_PATH="${PROJECT_PATH}/camera_calibration/rectified/masks"



# Output
OUTPUT_PATH="${PROJECT_PATH}/hierarchical-3dgs/output"
# SCAFFOLD_FILE="${OUTPUT_PATH}/scaffold/point_cloud/iteration_30000"
MODEL_PATH="${OUTPUT_PATH}/colmap_silvr" # Replace with the path to save the model

# Parameters
SAVE_ITERATIONS="10_000 30_000"
CHECKPOINT_ITERATIONS="30_000" # .chkpt 

# START_CHECKPOINT="$MODEL_PATH/chkpnt10000.pth"

# Run the Python script
python -u train_single.py \
    -s "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    -i "$IMAGES_PATH" \
    -d "$DEPTHS_PATH" \
    --skybox_locked \
    --save_iterations $SAVE_ITERATIONS \
    --checkpoint_iterations $CHECKPOINT_ITERATIONS \
    --dataset_type "colmap" \
    # --start_checkpoint "$START_CHECKPOINT"

    # --scaffold_file "$SCAFFOLD_FILE" \
