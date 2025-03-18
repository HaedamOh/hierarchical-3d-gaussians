#!/bin/bash
PROJECT_PATH="/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/raw"  


# Output Chunks
OUTPUT_CHUNKS="${PROJECT_PATH}/output/chunks"

# Source chunk
SOURCE_CHUNKS="${PROJECT_PATH}/camera_calibration/chunks"

# Output merged hierarchy
OUTPUT_MERGED_HIER="${PROJECT_PATH}/output/chunks/merged_hierarchy.hier"

# Chunk names
CHUNK_NAME_LIST="0_0 1_0"  


# Run the GaussianHierarchyCreator
/home/hierarchical-3d-gaussians/submodules/gaussianhierarchy/build/GaussianHierarchyMerger \
    "$OUTPUT_CHUNKS" \
    "0" \
    "$SOURCE_CHUNKS" \
    "$OUTPUT_MERGED_HIER" \
    $CHUNK_NAME_LIST 







