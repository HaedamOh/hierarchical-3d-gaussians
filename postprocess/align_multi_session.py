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
import os, sys, shutil
import numpy as np
import argparse
from read_write_model import *
import torch
import argparse
import os, time
from scipy import spatial
import json



def save_array_as_pcd(numpy_arr, filename="poses.pcd"):
    import open3d as o3d
    import torch
    pcd = o3d.geometry.PointCloud()
    pcd.points = o3d.utility.Vector3dVector(numpy_arr)
    o3d.io.write_point_cloud(filename, pcd)

def rotate_camera(qvec, tvec, rot_matrix, upscale,translation):
    # Assuming cameras have 'T' (translation) field

    R = qvec2rotmat(qvec)
    T = np.array(tvec)

    Rt = np.zeros((4, 4))
    Rt[:3, :3] = R
    Rt[:3, 3] = T
    Rt[3, 3] = 1.0

    C2W = np.linalg.inv(Rt)
    cam_center = np.copy(C2W[:3, 3])
    cam_rot_orig = np.copy(C2W[:3, :3])
    cam_center = np.matmul(cam_center, rot_matrix)
    cam_rot = np.linalg.inv(rot_matrix) @ cam_rot_orig 
    C2W[:3, 3] = upscale * cam_center + translation
    C2W[:3, :3] = cam_rot
    Rt = np.linalg.inv(C2W)
    new_pos = Rt[:3, 3]
    new_rot = rotmat2qvec(Rt[:3, :3])

    # R_test = qvec2rotmat(new_rots[-1])
    # T_test = np.array(new_poss[-1])
    # Rttest = np.zeros((4, 4))
    # Rttest[:3, :3] = R_test
    # Rttest[:3, 3] = T_test
    # Rttest[3, 3] = 1.0
    # C2Wtest = np.linalg.inv(Rttest) 

    return new_pos, new_rot

if __name__ == '__main__':

    parser = argparse.ArgumentParser(description='Automatically reorient colmap')    
    parser.add_argument('--input_path', type=str, help='Path to input colmap dir',  default='/home/shared/frontier_data/fnt_802/2024-12-13-10-36-rec002/processed/output_colmap/camera_calibration/aligned/sparse/0')
    parser.add_argument('--output_path', type=str, help='Path to output colmap dir',  default='/home/shared/frontier_data/fnt_802/2024-12-13-10-36-rec002/processed/output_colmap/camera_calibration/aligned_evo/sparse/0')
    parser.add_argument('--model_type', type=str, help='Specify which file format to use when processing colmap files (txt or bin)', choices=['bin','txt'], default='bin')
    parser.add_argument('--evo_alignment_json', type=str, default='/home/shared/frontier_data/fnt_802/2024-12-13-10-36-rec002/processed/output_colmap/camera_calibration/aligned/evo_align_results.json')
    args = parser.parse_args()

    if os.path.exists(args.output_path):
        confirmation = input("Are you sure you want to remove the existing COLMAP output? (yes/no): ").strip().lower()
        if confirmation == "yes":
            print("Removing existing COLMAP output...")
            shutil.rmtree(args.output_path)
        else:
            print("Operation cancelled.")

    # ** transformation to metric scale with evo ** # 
    with open(args.evo_alignment_json, 'r') as f:
        evo_align = json.load(f)
        scale = evo_align['scale']
        rotation = np.array(evo_align['rotation']).reshape(3, 3)
        translation = np.array(evo_align['translation'])  
        metric_transform = np.eye(4)
        metric_transform[:3, :3] = rotation * scale 
        metric_transform[:3, 3] = translation
        print(f"scale: {scale} \nrotation: {rotation} \ntranslation: {translation}")

    # Read colmap cameras, images and points
    cameras, images_metas_in, points3d_in = read_model(args.input_path, ext=f".{args.model_type}")
    positions = []
    print("Doing points")
    for key in points3d_in: 
        positions.append(points3d_in[key].xyz)
    
    positions = torch.from_numpy(np.array(positions))
    ## ** Points3D in metric scale saved in points3D.bin ## 
    rotated_points = scale * torch.matmul(positions, torch.from_numpy(rotation)) + torch.from_numpy(translation)
    save_array_as_pcd(rotated_points.numpy(), "debug_pcd/rotated_points_evo.pcd")
    points3d_out = {}
    for key, rotated in zip(points3d_in, rotated_points):
        point3d_in = points3d_in[key]
        points3d_out[key] = Point3D(
            id=point3d_in.id,
            xyz=rotated,
            rgb=point3d_in.rgb,
            error=point3d_in.error,
            image_ids=point3d_in.image_ids,
            point2D_idxs=point3d_in.point2D_idxs,
        )

    poses_pcd = []
    ## ** Images in metric scale saved in images.bin ## 
    images_metas_out = {} 
    for key in images_metas_in: 
        image_meta_in = images_metas_in[key]
        new_pos, new_rot = rotate_camera(image_meta_in.qvec, image_meta_in.tvec, rotation, scale, translation)
        W2C = np.eye(4)
        W2C[:3, :3] = qvec2rotmat(new_rot)
        W2C[:3, 3] = new_pos
        C2W = np.linalg.inv(W2C)     
        poses_pcd.append(C2W[:3, 3])
        
        images_metas_out[key] = Image(
            id=image_meta_in.id,
            qvec=new_rot,
            tvec=new_pos,
            camera_id=image_meta_in.camera_id,
            name=image_meta_in.name,
            xys=image_meta_in.xys,
            point3D_ids=image_meta_in.point3D_ids,
        )
    save_array_as_pcd(np.array(poses_pcd), "debug_pcd/poses_evo.pcd")
    if not os.path.isdir(args.output_path):
        os.makedirs(args.output_path)
        
    # Write the new model .bin
    write_model(cameras, images_metas_out, points3d_out, args.output_path, f".{args.model_type}")
    print(f"Saved scaled COLMAP to {args.output_path}") 

