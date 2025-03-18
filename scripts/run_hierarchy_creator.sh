#!/bin/bash
PROJECT_PATH="/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/raw"  
CHUNK_NAME="1_0"  

# Output Chunk
OUTPUT_CHUNK="${PROJECT_PATH}/output/chunks/${CHUNK_NAME}"
OUTPUT_CHUNK_POINT_CLOUD="${PROJECT_PATH}/output/chunks/${CHUNK_NAME}/point_cloud/iteration_30000/point_cloud.ply"

# Source chunk
SOURCE_CHUNK="${PROJECT_PATH}/camera_calibration/chunks/${CHUNK_NAME}"

# Scaffold
SCAFFOLD="${PROJECT_PATH}/output/scaffold/point_cloud/iteration_30000"

# Run the GaussianHierarchyCreator
/home/hierarchical-3d-gaussians/submodules/gaussianhierarchy/build/GaussianHierarchyCreator \
    "$OUTPUT_CHUNK_POINT_CLOUD" \
    "$SOURCE_CHUNK" \
    "$OUTPUT_CHUNK" \
    "$SCAFFOLD"







