#!/bin/bash
## Default COLMAP training chunk script ##


# Paths and parameters
CHUNK_NAME="0_0"  # Replace with the actual chunk name
PROJECT_PATH="/home/shared/frontier_data/fnt_802/2024-12-13-10-36-rec002/hierarchical-3dgs"  # Replace with the base path to your project

# Input
IMAGES_PATH="${PROJECT_PATH}/camera_calibration/rectified/images"
# DEPTHS_PATH="${PROJECT_PATH}/camera_calibration/rectified/depths"
# ALPHA_MASKS_PATH="${PROJECT_PATH}/camera_calibration/rectified/masks"

# Chunks
CHUNK_PATH="${PROJECT_PATH}/camera_calibration/chunks/${CHUNK_NAME}"
BOUNDS_FILE="${CHUNK_PATH}"

# Output
OUTPUT_PATH="${PROJECT_PATH}/output"
SCAFFOLD_FILE="${OUTPUT_PATH}/scaffold/point_cloud/iteration_30000"
MODEL_PATH="${OUTPUT_PATH}/chunks/${CHUNK_NAME}_test" # Replace with the path to save the model
# Parameters
SAVE_ITERATIONS="10_000 30_000"
CHECKPOINT_ITERATIONS="30_000" # .chkpt 

# START_CHECKPOINT="$MODEL_PATH/chkpnt10000.pth"

# Run the Python script
python -u train_single.py \
    -s "$CHUNK_PATH" \
    --model_path "$MODEL_PATH" \
    -i "$IMAGES_PATH" \
    --skybox_locked \
    --bounds_file "$BOUNDS_FILE" \
    --save_iterations $SAVE_ITERATIONS \
    --checkpoint_iterations $CHECKPOINT_ITERATIONS \
    --dataset_type "colmap" \
    # --eval \
    # --start_checkpoint "$START_CHECKPOINT"

    # --scaffold_file "$SCAFFOLD_FILE" \
    # -d "$DEPTHS_PATH" \
