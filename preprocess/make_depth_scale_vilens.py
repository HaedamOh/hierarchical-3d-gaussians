#
# Copyright (C) 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import numpy as np
import argparse
import cv2
from joblib import delayed, Parallel
import json
from read_write_model import *

import json
import numpy as np
import open3d as o3d
import cv2
from scipy.spatial.transform import Rotation as R
import os
import sys
sys.path.append(os.path.join(os.path.dirname(__file__), ".."))

from vilens_utils.dataset import FrontierDataset

def get_scales_custom(cam_pose, cam_intrinsic, cam_path,cam_label, args):
    """
    Compute scale and offset between a custom monocular depth map and 3D points from a PCD file.
    
    Args:
        cam_pose (dict): Camera pose containing rotation and translation.
        cam_intrinsic (dict): Camera intrinsic parameters.
        cam_path (Path): Path to the image.
        args: Additional arguments including paths.
    
    Returns:
        dict: Contains image name, scale, and offset.
    """
    # Load 3D points from Open3D
    pcd = o3d.io.read_point_cloud( os.path.join(args.vilens_slam_dir,args.points3d_pcd))
    points3d = np.asarray(pcd.points)  # Convert to NumPy array
    
    # Transform points using the provided extrinsics
    # cam_pose T_world_cam 
    # points3d T_world_point
    # T_cam_point = T_cam_world * T_world_point
    T_cam_world = np.linalg.inv(cam_pose)
    T_cam_point = np.dot(T_cam_world, np.hstack((points3d, np.ones((points3d.shape[0], 1)))).T).T
    pts = T_cam_point[:, :3]
    

    # Compute inverse depth
    invcolmapdepth = 1.0 / pts[:, 2]
    
    # Load monocular inverse depth map
    image_name = cam_path.stem
    invmonodepthmap = cv2.imread(f"{args.depths_dir}/{cam_label}/{image_name}.png", cv2.IMREAD_UNCHANGED)
    
    if invmonodepthmap is None:
        return None
    
    if invmonodepthmap.ndim != 2:
        invmonodepthmap = invmonodepthmap[..., 0]
    
    invmonodepthmap = invmonodepthmap.astype(np.float32) / (2**16)
    s = invmonodepthmap.shape[0] / cam_intrinsic.image_height
    breakpoint()
    # valid = 
    
    
    # Compute scale and offset if valid depths exist
    if (invcolmapdepth > 0).sum() > 10 and (invcolmapdepth.max() - invcolmapdepth.min()) > 1e-3:
        t_colmap = np.median(invcolmapdepth)
        s_colmap = np.mean(np.abs(invcolmapdepth - t_colmap))
        
        t_mono = np.median(invmonodepthmap)
        s_mono = np.mean(np.abs(invmonodepthmap - t_mono))
        scale = s_colmap / s_mono
        offset = t_colmap - t_mono * scale
    else:
        scale, offset = 0, 0
    print(f"Image: {image_name}, Scale: {scale}, Offset: {offset}")
    return {"image_name": image_name, "scale": scale, "offset": offset}


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--vilens_slam_dir', default='/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/vilens_slam', help='Path to JSON file containing camera intrinsics')
    parser.add_argument('--sensor_config',default='sensor_config/frn802/sensors_2024_08_07/sensors_frn802.yaml', help='Path to the sensor configuration file')
    parser.add_argument('--points3d_pcd', default='points3D_scaled.ply', help='Path to .pcd file with 3D points')
    parser.add_argument('--depths_dir', default='/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/vilens_slam/depths_rectified', help='Directory containing monocular depth maps')
    args = parser.parse_args()
    
    dataset = FrontierDataset(
        args.vilens_slam_dir,
        args.sensor_config,
        image_folder='images_rectified',
        slam_poses_csv="slam_poses.csv",
        image_poses_csv="image_poses.csv",
        slam_pose_graph_slam="slam_pose_graph.slam",
        slam_clouds_folder="slam_clouds",
        slam_pose_downsample_to=-1,
        )
    
    
    camera_poses, camera_paths, camera_timestamps = dataset.get_all_camera_poses(return_dict=True, sync_with_images=False, visualize=False)

    
    # depth_param_list = Parallel(n_jobs=-1, backend="threading")(
    #     delayed(get_scales_custom)(pose, dataset.sensor.get_sensor_param("camera", label), path,label, args)
    #     for label, poses in camera_poses.items()
    #     for pose, path in zip(poses, camera_paths[label])
    # )
    
    depth_param_list = [get_scales_custom(pose, dataset.sensor.get_sensor_param("camera", label), path,label, args) for label, poses in camera_poses.items() for pose, path in zip(poses, camera_paths[label])] 
    
    
    depth_params = {
        depth_param["image_name"]: {"scale": depth_param["scale"], "offset": depth_param["offset"]}
        for depth_param in depth_param_list if depth_param is not None
    }
    
    # Save results to JSON
    with open(os.path.join(args.vilens_slam_dir,'depth_params.json') , "w") as f:
        json.dump(depth_params, f, indent=2)
    
    print("Processing complete. Scale and offset values saved.")