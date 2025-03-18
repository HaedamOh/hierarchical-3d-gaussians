#!/bin/bash

# Paths and parameters
PROJECT_PATH="/home/shared/nerfstudio/oxford_spires/2024-03-13-maths_2"
SOURCE_PATH="${PROJECT_PATH}" 
IMAGES_PATH="${PROJECT_PATH}"
DEPTHS_PATH="${PROJECT_PATH}"

TRANSFORMS_JSON="${PROJECT_PATH}/submaps_rect/submap_1.json"
POINTCLOUD_FILE="${PROJECT_PATH}/submaps_rect/submap_1/cloud_lidar.pcd"

# Output
OUTPUT_PATH="${PROJECT_PATH}/output"
MODEL_PATH="${OUTPUT_PATH}/h-3dgs/submap_1_wo_depth" # Replace with the path to save the model
SCAFFOLD_FILE="${OUTPUT_PATH}/scaffold/point_cloud/iteration_30000"

# Parameters
SAVE_ITERATIONS="10_000 20_000"


# Run the Python script
CUDA_VISIBLE_DEVICES=1 python train_single.py \
    -s "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --images "$IMAGES_PATH" \
    --skybox_locked \
    --save_iterations $SAVE_ITERATIONS \
    --scaffold_file "$SCAFFOLD_FILE" \
    --dataset_type "silvr" \
    --transforms_json "$TRANSFORMS_JSON" \
    --pointcloud_file "$POINTCLOUD_FILE" \
    # --depths "$DEPTHS_PATH"

# Optional arguments (uncomment if needed)
# --depths "$DEPTHS_PATH" \
# --scaffold_file "$SCAFFOLD_FILE"