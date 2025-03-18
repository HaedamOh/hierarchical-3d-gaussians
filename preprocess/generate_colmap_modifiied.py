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
import subprocess
import argparse
from read_write_model import read_images_binary,write_images_binary, Image
import time, platform
from dataclasses import dataclass
from pathlib import Path
from typing import Union
import sys 
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
from vilens_utils.sensor import Sensor
from dataclasses import dataclass, field



yaml_path = Path(os.path.join(os.path.dirname(__file__),"../sensor_config/frn802/sensors_2024_08_07/sensors_frn802.yaml"))
sensor = Sensor.load_config(yaml_path)
@dataclass
class COLMAPConfig:
    # Incorporate the sensor config into COLMAPConfig.
    sensor: any = field(default=None, init=True, repr=True)
    matcher_method: str = "vocab_tree_matcher"
    sift_max_num_features: int = 1000
    sfm_ba_global_function_tolerance: float = 0
    sfm_max_refinements: int = 5
    sfm_max_iterations: int = 50
    ba_max_num_iterations: int = 100
    ba_max_linear_solver_iterations: int = 200
    ba_global_function_tolerance: float = 1e-6
    sequential_matcher_loop_detection_period: int = 10
    use_gpu: bool = True
    
    # To remove existing colmap output
    remove_existing_colmap_output: bool = True
    
    # Feature matching 
    use_prior_intrinsics: bool = True
    # Bundle Adjustment
    fix_focal_length: bool = True
    fix_principal_point: bool = True
    fix_extra_params: bool = True
    prior_img_width: int = 1920
    prior_img_height: int = 1080

colmap_config = COLMAPConfig(sensor)




def replace_images_by_masks(images_file, out_file):
    """Replace images.jpg to images.png in the colmap images.bin to process masks the same way as images."""
    images_metas = read_images_binary(images_file)
    out_images_metas = {}
    for key in images_metas:
        in_image_meta = images_metas[key]
        out_images_metas[key] = Image(
            id=key,
            qvec=in_image_meta.qvec,
            tvec=in_image_meta.tvec,
            camera_id=in_image_meta.camera_id,
            name=in_image_meta.name[:-3]+"png",
            xys=in_image_meta.xys,
            point3D_ids=in_image_meta.point3D_ids,
        )
    
    write_images_binary(out_images_metas, out_file)

def setup_dirs(project_dir):
    """Create the directories that will be required.
    'project_dir' is the base directory 
    '/camera_calibration' will be colmap processed
    """
    if not os.path.exists(project_dir):
        print("creating project dir.")
        os.makedirs(project_dir)
    
    if not os.path.exists(os.path.join(project_dir, "camera_calibration/aligned")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/aligned/sparse/0"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/rectified")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/rectified"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/unrectified")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified"))
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified", "sparse"))

    if not os.path.exists(os.path.join(project_dir, "camera_calibration/unrectified", "sparse")):
        os.makedirs(os.path.join(project_dir, "camera_calibration/unrectified", "sparse"))

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--project_dir', type=str,default='/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/hierarchical-3dgs' )
    parser.add_argument('--images_dir', default="", help="Will be set to project_dir/inputs/images if not set")
    parser.add_argument('--masks_dir', default="", help="Will be set to project_dir/inputs/masks if exists and not set")
    args = parser.parse_args()
    
    if args.images_dir == "":
        args.images_dir = os.path.join(args.project_dir, "inputs/images")
    if args.masks_dir == "":
        args.masks_dir = os.path.join(args.project_dir, "inputs/masks")
        args.masks_dir = args.masks_dir if os.path.exists(args.masks_dir) else ""

    colmap_exe = "colmap.bat" if platform.system() == "Windows" else "colmap"
    start_time = time.time()

    print(f"Project will be built here ${args.project_dir} base images are available there ${args.images_dir}.")

    if colmap_config.remove_existing_colmap_output:
        confirmation = input("Are you sure you want to remove the existing COLMAP output? (yes/no): ").strip().lower()
        if confirmation == "yes":
            print("Removing existing COLMAP output...")
            shutil.rmtree(f"{args.project_dir}/camera_calibration", ignore_errors=True)
        else:
            print("Operation cancelled.")

    setup_dirs(args.project_dir)

        

    ## 1.1 Feature extraction, matching then mapper to generate the colmap. ( ~ 1min )
    print("[1]. Extracting features ...")
    if not colmap_config.use_prior_intrinsics:
        colmap_feature_extractor_args = [
            colmap_exe, "feature_extractor",
            "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
            "--image_path", f"{args.images_dir}",
            "--ImageReader.single_camera_per_folder 1",
            "--ImageReader.default_focal_length_factor", "0.5",
            "--ImageReader.camera_model", "OPENCV_FISHEYE",
            "--SiftExtraction.use_gpu", "1",
            ]
        
        try:
            subprocess.run(colmap_feature_extractor_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing colmap feature_extractor: {e}")
            sys.exit(1)
        
    # 1.1 Use prior intrinsics from Sensor (Frontier):
    if colmap_config.use_prior_intrinsics:
        confirmation = input("Are you sure you want to use VILENS intrinsics (yes/no): ").strip().lower()
        if confirmation != "yes":
            print("Operation cancelled.")
            sys.exit(0)
        print("Using sensor intrinsics for each camera")
        for camera in colmap_config.sensor.cameras:
            # Extract intrinsics for this specific camera
            K = camera.get_K(fisheye=True)
            D = camera.fisheye_extra_params
            fx, fy, cx, cy = K[0, 0], K[1, 1], K[0, 2], K[1, 2] 
            colmap_cam_params = f"{fx},{fy},{cx},{cy},{D[0]},{D[1]},{D[2]},{D[3]}"

            # COLMAP camera prior arguments for this camera
            colmap_camera_prior_args = [
                colmap_exe, "feature_extractor",
                "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
                "--image_path", f"{args.images_dir}",
                "--ImageReader.single_camera", "0",  # Set to 0 since multiple cameras exist
                "--ImageReader.default_focal_length_factor", "0.5",
                "--ImageReader.camera_model", "OPENCV_FISHEYE",
                "--SiftExtraction.use_gpu", "1",
            ]
            colmap_camera_prior_args += ["--ImageReader.camera_params", colmap_cam_params]
            
            print(f"Running COLMAP for camera_label:{camera.label} with camera_params: {colmap_cam_params}")

            # Run COLMAP for this camera
            try:
                subprocess.run(colmap_camera_prior_args, check=True)
            except subprocess.CalledProcessError as e:
                print(f"Error executing COLMAP for {camera.label}: {e}")
                sys.exit(1)

        print("COLMAP camera_prior completed for all cameras.")
        

    # 2.0 Custom macher (~ 3min)
    print("[2.0] Making custom matches...")
    make_colmap_custom_matcher_args = [
        "python", f"preprocess/make_colmap_custom_matcher.py",
        "--image_path", f"{args.images_dir}",
        "--output_path", f"{args.project_dir}/camera_calibration/unrectified/matching.txt"
    ]
    try:
        subprocess.run(make_colmap_custom_matcher_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing make_colmap_custom_matcher: {e}")
        sys.exit(1)

    ## 2.1 Feature matching
    print("[2.1] Matching features...")
    colmap_matches_importer_args = [
        colmap_exe, "matches_importer",
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--match_list_path", f"{args.project_dir}/camera_calibration/unrectified/matching.txt"
        ]
    try:
        subprocess.run(colmap_matches_importer_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap matches_importer: {e}")
        sys.exit(1)

    ## 3. Mapper (Bundle Admustment), Generate sfm pointcloud (~ 15min)
    print("[3] Mapper, generating sfm point cloud...")
    colmap_hierarchical_mapper_args = [
        colmap_exe, "hierarchical_mapper",
        "--database_path", f"{args.project_dir}/camera_calibration/unrectified/database.db",
        "--image_path", f"{args.images_dir}",
        "--output_path", f"{args.project_dir}/camera_calibration/unrectified/sparse",
        "--Mapper.ba_global_function_tolerance", "0.000001" 
        ]
    colmap_hierarchical_mapper_args += ["--Mapper.ba_local_max_num_iterations", "10"]
    
    
    if colmap_config.fix_focal_length:
        colmap_hierarchical_mapper_args += ["--Mapper.ba_refine_focal_length", "0"]
    if colmap_config.fix_principal_point:
        colmap_hierarchical_mapper_args += ["--Mapper.ba_refine_principal_point", "0"]
    if colmap_config.fix_extra_params:
        colmap_hierarchical_mapper_args += ["--Mapper.ba_refine_extra_params", "0"]
    
    try:
        subprocess.run(colmap_hierarchical_mapper_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing colmap hierarchical_mapper: {e}")
        sys.exit(1)

    ## Simplify images so that everything takes less time (reading colmap usually takes forever)
    simplify_images_args = [
        "python", f"preprocess/simplify_images.py",
        "--base_dir", f"{args.project_dir}/camera_calibration/unrectified/sparse/0"
    ]
    try:
        subprocess.run(simplify_images_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing simplify_images: {e}")
        sys.exit(1)

    ## 4. Undistort images 
    print(f"undistorting images from {args.images_dir} to {args.project_dir}/camera_calibration/rectified images...")
    colmap_image_undistorter_args = [
        colmap_exe, "image_undistorter",
        "--image_path", f"{args.images_dir}",
        "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0", 
        "--output_path", f"{args.project_dir}/camera_calibration/rectified/",
        "--output_type", "COLMAP",
        "--max_image_size", f"{max(COLMAPConfig.prior_img_width, COLMAPConfig.prior_img_height)}",
        ]
    try:
        subprocess.run(colmap_image_undistorter_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing image_undistorter: {e}")
        sys.exit(1)

    if not args.masks_dir == "":
        # create a copy of colmap as txt and replace jpgs with pngs to undistort masks the same way images were distorted
        if not os.path.exists(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks"):
            os.makedirs(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks")

        shutil.copy(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/cameras.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/cameras.bin")
        shutil.copy(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/points3D.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/points3D.bin")
        replace_images_by_masks(f"{args.project_dir}/camera_calibration/unrectified/sparse/0/images.bin", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks/images.bin")

        print("undistorting masks aswell...")
        colmap_image_undistorter_args = [
            colmap_exe, "image_undistorter",
            "--image_path", f"{args.masks_dir}",
            "--input_path", f"{args.project_dir}/camera_calibration/unrectified/sparse/0/masks", 
            "--output_path", f"{args.project_dir}/camera_calibration/tmp/",
            "--output_type", "COLMAP",
            "--max_image_size", f"{max(COLMAPConfig.prior_img_width, COLMAPConfig.prior_img_height)}",
            ]
        try:
            subprocess.run(colmap_image_undistorter_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing image_undistorter: {e}")
            sys.exit(1)
        
        make_mask_uint8_args = [
            "python", f"preprocess/make_mask_uint8.py",
            "--in_dir", f"{args.project_dir}/camera_calibration/tmp/images",
            "--out_dir", f"{args.project_dir}/camera_calibration/rectified/masks"
        ]
        try:
            subprocess.run(make_mask_uint8_args, check=True)
        except subprocess.CalledProcessError as e:
            print(f"Error executing make_colmap_custom_matcher: {e}")
            sys.exit(1)

        # remove temporary dir containing undistorted masks
        shutil.rmtree(f"{args.project_dir}/camera_calibration/tmp")

    # 5. re-orient + scale colmap
    print(f"re-orient and scaling scene to {args.project_dir}/camera_calibration/aligned/sparse/0")
    reorient_args = [
            "python", f"preprocess/auto_reorient.py",
            "--input_path", f"{args.project_dir}/camera_calibration/rectified/sparse",
            "--output_path", f"{args.project_dir}/camera_calibration/aligned/sparse/0"
        ]
    try:
        subprocess.run(reorient_args, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Error executing auto_orient: {e}")
        sys.exit(1)

    end_time = time.time()
    print(f"Preprocessing done in {(end_time - start_time)/60.0} minutes.")
