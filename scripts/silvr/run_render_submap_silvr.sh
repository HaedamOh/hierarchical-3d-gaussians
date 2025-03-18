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

# HIERACHY_PATH="${OUTPUT_PATH}/chunks/merged_hierarchy.hier"
OUT_DIR="${MODEL_PATH}/render"  


# Run the Python script
python -u render_single.py \
    -s "$SOURCE_PATH" \
    --model_path "$MODEL_PATH" \
    --images "$IMAGES_PATH" \
    --depths "$DEPTHS_PATH" \
    --dataset_type "silvr" \
    --out_dir "$OUT_DIR" \
    --transforms_json "$TRANSFORMS_JSON" \
    --pointcloud_file "$POINTCLOUD_FILE" \
