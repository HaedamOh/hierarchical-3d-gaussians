#
# Copyright (C) 2023 - 2024, Inria
# GRAPHDECO research group, https://team.inria.fr/graphdeco
# All rights reserved.
#
# This software is free for non-commercial, research and evaluation use 
# under the terms of the LICENSE.md file.
#
# For inquiries contact  george.drettakis@inria.fr
#

import os
import sys
from PIL import Image
from typing import NamedTuple
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text, qvec2rotmat, \
    read_extrinsics_binary, read_intrinsics_binary, read_points3D_binary, read_points3D_text
from utils.graphics_utils import getWorld2View2, focal2fov, fov2focal
import numpy as np
import json
from pathlib import Path
from plyfile import PlyData, PlyElement
from utils.sh_utils import SH2RGB
from scene.gaussian_model import BasicPointCloud
import torch

from vilens_utils.dataset import FrontierDataset

class CameraInfo(NamedTuple):
    uid: int
    R: np.array
    T: np.array
    FovY: np.array
    FovX: np.array
    primx:float
    primy:float
    depth_params: dict
    image_path: str
    mask_path: str
    depth_path: str
    image_name: str
    width: int
    height: int
    is_test: bool

class SceneInfo(NamedTuple):
    point_cloud: BasicPointCloud
    train_cameras: list
    test_cameras: list
    nerf_normalization: dict
    ply_path: str

def getNerfppNorm(cam_info):
    def get_center_and_diag(cam_centers):
        cam_centers = np.hstack(cam_centers)
        avg_cam_center = np.mean(cam_centers, axis=1, keepdims=True)
        center = avg_cam_center
        dist = np.linalg.norm(cam_centers - center, axis=0, keepdims=True)
        diagonal = np.quantile(dist, 0.9)
        return center.flatten(), diagonal

    cam_centers = []

    for cam in cam_info:
        W2C = getWorld2View2(cam.R, cam.T)
        C2W = np.linalg.inv(W2C)
        cam_centers.append(C2W[:3, 3:4])

    center, diagonal = get_center_and_diag(cam_centers)
    radius = diagonal * 1.1

    translate = -center
    print('Radius: ', radius, "Center: ", center )    
    return {"translate": translate, "radius": radius}

def readColmapCameras(cam_extrinsics, cam_intrinsics, depths_params, images_folder, masks_folder, depths_folder, test_cam_names_list):
    
    ##### TODO DEBUG: VILENS / COLMAP test ##########
    # dataset = FrontierDataset(
    #     slam_output_folder_path='/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/vilens_slam',
    #     sensor_config_yaml_path='./sensor_config/frn802/sensors_2024_08_07/sensors_frn802.yaml',
    #     image_folder='images_rectified',
    #     slam_poses_csv="slam_poses.csv",
    #     image_poses_csv="image_poses.csv",
    #     slam_pose_graph_slam="slam_pose_graph.slam",
    #     slam_clouds_folder="slam_clouds",
    #     slam_pose_downsample_to=-1,
    #     )
    # camera_poses, camera_paths, camera_timestamps = dataset.get_all_camera_poses(return_dict=True, sync_with_images=False, visualize=False)
    # idx = 0
    ##########################
    cam_infos = []
    for idx, key in enumerate(cam_extrinsics):
        sys.stdout.write('\r')
        # the exact output you're looking for:
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(cam_extrinsics)))
        sys.stdout.flush()

        extr = cam_extrinsics[key]
        intr = cam_intrinsics[extr.camera_id]
        height = intr.height
        width = intr.width

        uid = intr.id
        R = np.transpose(qvec2rotmat(extr.qvec))
        T = np.array(extr.tvec)

        if intr.model=="SIMPLE_PINHOLE":
            focal_length_x = intr.params[0]
            primx = float(intr.params[1]) / width
            primy = float(intr.params[2]) / height
            FovY = focal2fov(focal_length_x, height)
            FovX = focal2fov(focal_length_x, width)
        elif intr.model=="PINHOLE":
            primx = float(intr.params[2]) / width
            primy = float(intr.params[3]) / height
            focal_length_x = intr.params[0]
            focal_length_y = intr.params[1]
            FovY = focal2fov(focal_length_y, height)
            FovX = focal2fov(focal_length_x, width)
        else:
            assert False, "Colmap camera model not handled: only undistorted datasets (PINHOLE or SIMPLE_PINHOLE cameras) supported!"

        n_remove = len(extr.name.split('.')[-1]) + 1
        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[extr.name[:-n_remove]]
            except:
                print("\n", key, "not found in depths_params")

        image_path = os.path.join(images_folder, extr.name)
        image_name = extr.name
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{extr.name[:-n_remove]}.jpg")
            image_name = f"{extr.name[:-n_remove]}.jpg"
        if not os.path.exists(image_path):
            image_path = os.path.join(images_folder, f"{extr.name[:-n_remove]}.png")
            image_name = f"{extr.name[:-n_remove]}.png"

        mask_path = os.path.join(masks_folder, f"{extr.name[:-n_remove]}.png") if masks_folder != "" else ""
        depth_path = os.path.join(depths_folder, f"{extr.name[:-n_remove]}.png") if depths_folder != "" else ""    
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, primx=primx, primy=primy, depth_params=depth_params,
                              image_path=image_path, mask_path=mask_path, depth_path=depth_path, image_name=image_name, 
                              width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def fetchPly(path):
    plydata = PlyData.read(path)
    vertices = plydata['vertex']
    positions = np.vstack([vertices['x'], vertices['y'], vertices['z']]).T

    if('red' in vertices):
        colors = np.vstack([vertices['red'], vertices['green'], vertices['blue']]).T / 255.0
    else:
        colors = np.ones_like(positions) * 0.5
    if('nx' in vertices):
        normals = np.vstack([vertices['nx'], vertices['ny'], vertices['nz']]).T
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPcd(path):
    import open3d as o3d
    pcd = o3d.io.read_point_cloud(path)
    positions = np.asarray(pcd.points)
    if pcd.has_colors():
        colors = np.asarray(pcd.colors)
    else:
        colors = np.ones_like(positions) * 0.5
    if pcd.has_normals():
        normals = np.asarray(pcd.normals)
    else:
        normals = np.zeros_like(positions)
    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def fetchPt(xyz_path, rgb_path):
    positions_tensor = torch.jit.load(xyz_path).state_dict()['0']

    positions = positions_tensor.numpy()

    colors_tensor = torch.jit.load(rgb_path).state_dict()['0']
    if colors_tensor.size(0) == 0:
        colors_tensor = 255 * (torch.ones_like(positions_tensor) * 0.5)
    colors = (colors_tensor.float().numpy()) / 255.0
    normals = torch.Tensor([]).numpy()

    return BasicPointCloud(points=positions, colors=colors, normals=normals)

def storePly(path, xyz, rgb):
    # Define the dtype for the structured array
    dtype = [('x', 'f4'), ('y', 'f4'), ('z', 'f4'),
            ('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4'),
            ('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    
    normals = np.zeros_like(xyz)

    elements = np.empty(xyz.shape[0], dtype=dtype)
    attributes = np.concatenate((xyz, normals, rgb), axis=1)
    elements[:] = list(map(tuple, attributes))

    # Create the PlyData object and write to file
    vertex_element = PlyElement.describe(elements, 'vertex')
    ply_data = PlyData([vertex_element])
    ply_data.write(path)

def readColmapSceneInfo(path, images, masks, depths, eval, train_test_exp, llffhold=20):
    try:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.bin")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.bin")
        cam_extrinsics = read_extrinsics_binary(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_binary(cameras_intrinsic_file)
    except:
        cameras_extrinsic_file = os.path.join(path, "sparse/0", "images.txt")
        cameras_intrinsic_file = os.path.join(path, "sparse/0", "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)

    ## ** Debug if use scaled depth params ** ## 
    depth_params_file = os.path.join(path, "sparse/0", "depth_params.json")
    # depth_params_file = os.path.join(path, "sparse/0", "depth_params_scaled.json")
    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)


    ply_path = os.path.join(path, "sparse/0/points3D.ply")
    bin_path = os.path.join(path, "sparse/0/points3D.bin")
    txt_path = os.path.join(path, "sparse/0/points3D.txt")
    # # ** Debug if use scaled point cloud ** ## 
    # ply_path = os.path.join(path, "sparse/0/points3D_scaled.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D_scaled.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D_scaled.txt")
    
    try:
        xyz_path = os.path.join(path, "sparse/0/xyz.pt")
        rgb_path = os.path.join(path, "sparse/0/rgb.pt")
        pcd = fetchPt(xyz_path, rgb_path)
    except:
        if not os.path.exists(ply_path):
            print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
            try:
                xyz, rgb, _ = read_points3D_binary(bin_path)
            except:
                xyz, rgb, _ = read_points3D_text(txt_path)
            storePly(ply_path, xyz, rgb)
        pcd = fetchPly(ply_path)

    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [cam_extrinsics[cam_id].name for cam_id in cam_extrinsics]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
            with open(os.path.join(path, "sparse/0", "test.txt"), 'w') as file:
                for line in test_cam_names_list:
                    file.write(line + '\n')
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
            print(len(test_cam_names_list), "test images")
    else:
        test_cam_names_list = []

    # reading_dir = "images" if images == None else images
    masks_reading_dir = masks if masks == "" else os.path.join(path, masks)

    cam_infos_unsorted = readColmapCameras(
        cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=depths_params, 
        images_folder=images, masks_folder=masks_reading_dir,
        depths_folder=depths if depths != "" else "", test_cam_names_list=test_cam_names_list)
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)

    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    print(len(test_cam_infos), "test images")
    print(len(train_cam_infos), "train images")

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                           train_cameras=train_cam_infos,
                           test_cameras=test_cam_infos,
                           nerf_normalization=nerf_normalization,
                           ply_path=ply_path)
    return scene_info


# SILVR dataset reader #
def readCamerasFromSILVRTransforms(contents, images, depths,depths_params, masks, test_cam_names_list):
    """
    Read transforms.json (graphics -> vision)
    """
    cam_infos = []
    frames = contents["frames"]
    for idx, frame in enumerate(frames):
        sys.stdout.write('\r')
        sys.stdout.write("Reading camera {}/{}".format(idx+1, len(frames)))
        sys.stdout.flush()
        
        T_WC = np.array(frame["transform_matrix"])
        c2w = T_WC @ PoseConvention.transforms["graphics"]["vision"]  # graphics -> vision
        w2c = np.linalg.inv(c2w)
        
        uid = idx
        R = np.transpose(w2c[:3, :3])
        T = w2c[:3, 3]

        fx, fy, cx, cy = frame["fl_x"], frame["fl_y"], frame["cx"], frame["cy"]
        width,height = frame["w"], frame["h"]
        
        image_path = os.path.join(images, frame["file_path"])
        image_name = Path(frame["file_path"]).parent.name + "/" + Path(frame["file_path"]).stem

        primx = cx / width
        primy = cy / height
        FovX = focal2fov(fx, width)
        FovY = focal2fov(fy, height)

        mask_path = os.path.join(masks, frame["mask_file_path"]) if masks != "" else ""
        depth_path = os.path.join(depths, frame["depth_file_path"]) if depths != "" else ""   

        depth_params = None
        if depths_params is not None:
            try:
                depth_params = depths_params[image_name]
            except:
                print("\n", frame["file_path"], "not found in depths_params")
        
        cam_info = CameraInfo(uid=uid, R=R, T=T, FovY=FovY, FovX=FovX, primx=primx, primy=primy, depth_params=depth_params,
                            image_path=image_path, mask_path=mask_path, depth_path=depth_path, image_name=image_name, 
                            width=width, height=height, is_test=image_name in test_cam_names_list)
        cam_infos.append(cam_info)
    sys.stdout.write('\n')
    return cam_infos

def readSilvrInfo(path, images, masks, depths, eval, train_test_exp,transforms_json, pointcloud_file, llffhold=8):
    print("[Reading SILVR]", transforms_json)
    with open(os.path.join(transforms_json)) as json_file:
        contents = json.load(json_file)
    
    depth_params_file = os.path.join(path, "depth_params.json")
    assert os.path.exists(depth_params_file), "depth_params.json file not found at path '{depth_params_file}'."
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)
    
    if pointcloud_file is not None:
        ply_path = pointcloud_file
        if ply_path.endswith(".pcd"):
            pcd = fetchPcd(ply_path)
        elif ply_path.endswith(".ply"):
            pcd = fetchPly(ply_path)
        else:
            print("Unsupported pointcloud file format")
            sys.exit(1)
    
    else:
        ply_path = os.path.join(path, "sparse/0/points3D.ply")
        bin_path = os.path.join(path, "sparse/0/points3D.bin")
        txt_path = os.path.join(path, "sparse/0/points3D.txt")   
        try:
            xyz_path = os.path.join(path, "sparse/0/xyz.pt")
            rgb_path = os.path.join(path, "sparse/0/rgb.pt")
            pcd = fetchPt(xyz_path, rgb_path)
        except:
            if not os.path.exists(ply_path):
                print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
                try:
                    xyz, rgb, _ = read_points3D_binary(bin_path)
                except:
                    xyz, rgb, _ = read_points3D_text(txt_path)
                storePly(ply_path, xyz, rgb)
            try:
                pcd = fetchPly(ply_path)
            except:
                pcd = fetchPcd(ply_path)
        
    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            cam_names = [frame['file_path'] for id,frame in enumerate(contents)]
            cam_names = sorted(cam_names)
            test_cam_names_list = [name for idx, name in enumerate(cam_names) if idx % llffhold == 0]
            with open(os.path.join(path, "sparse/0", "test.txt"), 'w') as file:
                for line in test_cam_names_list:
                    file.write(line + '\n')
        else:
            with open(os.path.join(path, "sparse/0", "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
            print(len(test_cam_names_list), "test images")
    else:
        test_cam_names_list = []


    cam_infos_unsorted = readCamerasFromSILVRTransforms(contents,images, depths, depths_params, masks, test_cam_names_list)
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    print(len(test_cam_infos), "test images")
    print(len(train_cam_infos), "train images")    
        
    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=train_cam_infos,
                        test_cameras=test_cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path)
    return scene_info




# VILENS folder reader, reading Frontier_dataset  #
def readCamerasFromVILENSFolder(dataset,camera_poses,camera_paths,camera_timestamps, depths, masks, test_cam_names_list ):
    """
    Read VILENS output folder
    """
    cam_infos = []
    idx = 0
    for camera_label, pose in camera_poses.items():
        cam_poses = camera_poses[camera_label]  # List of poses for this camera
        cam_paths = camera_paths[camera_label]  # List of paths for this camera
        cam_timestamps = camera_timestamps[camera_label]  # List of timestamps for this camera 
        cameras_params = dataset.sensor.get_sensor_param("camera", camera_label) 
        print(f"Reading {camera_label} :", len(cam_poses), "images")

        for cam_pose, cam_path, cam_timestamp in zip(cam_poses, cam_paths, cam_timestamps):
            c2w = cam_pose
            # get the world-to-camera transform: T_CW
            w2c = np.linalg.inv(c2w)
            R = np.transpose(w2c[:3, :3])  # R is stored transposed due to 'glm' in CUDA code
            T = w2c[:3, 3]

            image_path = str(cam_path)
            image_name = str( Path(cam_path.parent.name) / cam_path.name )   # cam0/image_00000_00000.jpg
            cam_intrinsics = cameras_params.rect_intrinsics
            fx, fy, cx, cy = (
                cam_intrinsics[0],
                cam_intrinsics[4],
                cam_intrinsics[2],
                cam_intrinsics[5],
            )
            width = cameras_params.image_width
            height = cameras_params.image_height
            primx = cx / width
            primy = cy / height
            FovX = focal2fov(fx, width)
            FovY = focal2fov(fy, height)
            
            mask_path = os.path.join(masks, f"{image_name}.png") if masks != "" else ""
            depth_path = os.path.join(depths, f"{image_name}.png") if depths != "" else ""
            
            cam_info = CameraInfo(uid=idx, R=R, T=T, FovY=FovY, FovX=FovX, primx=primx, primy=primy, depth_params=None,
                              image_path=image_path, mask_path=mask_path, depth_path=depth_path, image_name=image_name, 
                              width=width, height=height, is_test=image_name in test_cam_names_list)
            cam_infos.append(cam_info)
            idx += 1

    return cam_infos


# VILENS ROS2 format #
def readVilensInfo(path,images,masks, depths,eval,train_test_exp,llffhold=None):
    """
    Args: transforms_colmap_scaled.json (VILENS coordinate frame, scaled to meters)
    """
    ########################################################
    print("Reading VILENS_ROS2")
    sensor_config_path = './sensor_config/frn802/sensors_2024_08_07/sensors_frn802.yaml'
    depth_params_file = '/home/shared/frontier_data/fnt_802/2025-01-20_19-52-11-hbac-quad_fnt802/raw/camera_calibration/aligned/sparse/0/depth_params_scaled.json'  # This should also be scaled... 
    print('Hardcoded values: sensor_config_path, depth_params_file')
    ########################################################

    dataset = FrontierDataset(
        path,
        sensor_config_path,
        image_folder=images,
        slam_poses_csv="slam_poses.csv",
        image_poses_csv="image_poses.csv",
        slam_pose_graph_slam="slam_pose_graph.slam",
        slam_clouds_folder="slam_clouds",
        slam_pose_downsample_to=-1,
        )
    camera_poses, camera_paths, camera_timestamps = dataset.get_all_camera_poses(return_dict=True, sync_with_images=False, visualize=False)

    ## if depth_params_file isnt there AND depths file is here -> throw error
    depths_params = None
    if depths != "":
        try:
            with open(depth_params_file, "r") as f:
                depths_params = json.load(f)
            all_scales = np.array([depths_params[key]["scale"] for key in depths_params])
            if (all_scales > 0).sum():
                med_scale = np.median(all_scales[all_scales > 0])
            else:
                med_scale = 0
            for key in depths_params:
                depths_params[key]["med_scale"] = med_scale

        except FileNotFoundError:
            print(f"Error [SILVR]: depth_params.json file not found at path '{depth_params_file}'.")
            sys.exit(1)
        except Exception as e:
            print(f"An unexpected error occurred when trying to open depth_params.json file: {e}")
            sys.exit(1)

    # We use scaled one for silvr 
    ply_path = os.path.join(path, "slam_combined_cloud_small.pcd")
    pcd = fetchPcd(ply_path)

    # ply_path = os.path.join(path, "sparse/0/points3D_scaled.ply")
    # bin_path = os.path.join(path, "sparse/0/points3D_scaled.bin")
    # txt_path = os.path.join(path, "sparse/0/points3D_scaled.txt")

    
    # try:
    #     xyz_path = os.path.join(path, "sparse/0/xyz.pt")
    #     rgb_path = os.path.join(path, "sparse/0/rgb.pt")
    #     pcd = fetchPt(xyz_path, rgb_path)
    # except:
    #     if not os.path.exists(ply_path):
    #         print("Converting point3d.bin to .ply, will happen only the first time you open the scene.")
    #         try:
    #             xyz, rgb, _ = read_points3D_binary(bin_path)
    #         except:
    #             xyz, rgb, _ = read_points3D_text(txt_path)
    #         storePly(ply_path, xyz, rgb)
    #     pcd = fetchPly(ply_path)


    if eval:
        if "360" in path:
            llffhold = 8
        if llffhold:
            print("------------LLFF HOLD-------------")
            test_cam_names_list = [ str(Path(path.parent.name) / path.name) for paths in camera_paths.values() for idx, path in enumerate(paths) if idx % llffhold == 0]
            with open(os.path.join(path, "test.txt"), 'w') as file:
                for line in test_cam_names_list:
                    file.write(line + '\n')
        else:
            with open(os.path.join(path, "test.txt"), 'r') as file:
                test_cam_names_list = [line.strip() for line in file]
            print(len(test_cam_names_list), "test images")
    else:
        test_cam_names_list = []

    # reading_dir = "images" if images == None else images
    # masks_reading_dir = masks if masks == "" else os.path.join(path, masks)

    cam_infos_unsorted = readCamerasFromVILENSFolder(dataset,camera_poses,camera_paths,camera_timestamps, depths, masks, test_cam_names_list )
    print("Number of total cameras images: ", len(cam_infos_unsorted))
    
    cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
    train_cam_infos = [c for c in cam_infos if train_test_exp or not c.is_test]
    test_cam_infos = [c for c in cam_infos if c.is_test]
    print(len(test_cam_infos), "test images")
    print(len(train_cam_infos), "train images")   

    nerf_normalization = getNerfppNorm(train_cam_infos)

    scene_info = SceneInfo(point_cloud=pcd,
                        train_cameras=train_cam_infos,
                        test_cameras=test_cam_infos,
                        nerf_normalization=nerf_normalization,
                        ply_path=ply_path)
    return scene_info


sceneLoadTypeCallbacks = {
    "Colmap": readColmapSceneInfo,
    "Silvr": readSilvrInfo,
    'Vilens': readVilensInfo
}



class PoseConvention:
    """
    robotics (default)
    x forward, y left, z up

    computer vision / colmap
    x right, y down, z foward

    computer graphics / blender / nerf
    x right, y up, z backward
    """

    supported_conventions = ["robotics", "vision", "graphics"]
    graphics2robotics = np.array([[0, -1, 0, 0], [0, 0, 1, 0], [-1, 0, 0, 0], [0, 0, 0, 1]])
    # robotics2blender = blender2robotics.T
    vision2robotics = np.array([[0, -1, 0, 0], [0, 0, -1, 0], [1, 0, 0, 0], [0, 0, 0, 1]])
    # robotics2colmap = colmap2robotics.T
    vision2graphics = vision2robotics @ graphics2robotics.T
    # blender2colmap = colmap2blender.T

    # T_WB x T_BA, used to transform from B to A
    transforms = {
        "robotics": {"robotics": np.eye(4), "graphics": graphics2robotics.T, "vision": vision2robotics.T},
        "vision": {"robotics": vision2robotics, "graphics": vision2graphics, "vision": np.eye(4)},
        "graphics": {"robotics": graphics2robotics, "graphics": np.eye(4), "vision": vision2graphics.T},
    }







    @staticmethod
    def rename_convention(convention):
        if convention in ["nerf", "blender"]:
            convention = "graphics"
        elif convention in ["colmap"]:
            convention = "vision"
        assert convention in PoseConvention.supported_conventions, f"Unsupported convention: {convention}"
        return convention

    @staticmethod
    def get_transform(input_convention, output_convention):
        input_convention = PoseConvention.rename_convention(input_convention)
        output_convention = PoseConvention.rename_convention(output_convention)
        return PoseConvention.transforms[input_convention][output_convention]