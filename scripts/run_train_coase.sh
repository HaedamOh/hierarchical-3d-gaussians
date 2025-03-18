#!/bin/bash

# Paths and parameters
ALIGNED_PATH=/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/raw # '/home/docker_dev/hierarchical-3d-gaussians/example_dataset'
SKYBOX_NUM=100000


# Iteration values for save and checkpoint
SAVE_ITERATIONS="10_000 20_000 30_000"
CHECKPOINT_ITERATIONS="10_000 20_000 30_000"

POSITION_LR_INIT=0.00016
POSITION_LR_FINAL=0.0000016

# Run the Python script
python train_coarse.py \
    -s "${ALIGNED_PATH}/camera_calibration/aligned" \
    --skybox_num "$SKYBOX_NUM" \
    --position_lr_init "$POSITION_LR_INIT" \
    --position_lr_final "$POSITION_LR_FINAL" \
    -i "${ALIGNED_PATH}/camera_calibration/rectified/images" \
    --save_iterations $SAVE_ITERATIONS \
    --checkpoint_iterations $CHECKPOINT_ITERATIONS

