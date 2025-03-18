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

import json 
import math
from pathlib import Path 
import os
import torch
from random import randint
from utils.loss_utils import ssim
from gaussian_renderer import render
import sys
from scene import Scene, GaussianModel
from tqdm import tqdm
from utils.image_utils import psnr
from argparse import ArgumentParser
from arguments import ModelParams, PipelineParams, OptimizationParams
import torchvision
from lpipsPyTorch import lpips

from utils.camera_utils import JSON_to_camera
from scene.colmap_loader import read_extrinsics_text, read_intrinsics_text
from scene.dataset_readers import readColmapCameras
from utils.camera_utils import dummy_cameraList_from_camInfos 
def direct_collate(x):
    return x



@torch.no_grad()
def render_novel_view(args, scene, pipe, out_dir, tau, eval):
    render_path = out_dir

    # ** cameras.json file for training data ** 
    # camera_json_path = '/home/shared/frontier_data/fnt_802/2024-12-13-10-36-rec002/processed/output_colmap/output/aligned_evo/cameras.json'
    # with open(camera_json_path, 'r') as f:
    #     camera_json = json.load(f)
    # cameras = []
    # for cam_info in camera_json:
    #     camera = JSON_to_camera(cam_info)
    #     cameras.append(camera)
    
    # ** cameras.txt and images.txt files for novel viewpoints **
    if args.camera_novel_paths is not None:
        path = args.camera_novel_paths
        cameras_extrinsic_file = os.path.join(path, "images.txt")
        cameras_intrinsic_file = os.path.join(path, "cameras.txt")
        cam_extrinsics = read_extrinsics_text(cameras_extrinsic_file)
        cam_intrinsics = read_intrinsics_text(cameras_intrinsic_file)
        
        cam_infos_unsorted = readColmapCameras(
            cam_extrinsics=cam_extrinsics, cam_intrinsics=cam_intrinsics, depths_params=None, 
            images_folder=path, masks_folder='',
            depths_folder='', test_cam_names_list=[])
        
        cam_infos = sorted(cam_infos_unsorted.copy(), key = lambda x : x.image_name)
        cameras = dummy_cameraList_from_camInfos(cam_infos, args.resolution,  args.eval) 
    
    # cameras = scene.getTestCameras() if eval else scene.getTrainCameras()
    print(f"Rendering {len(cameras)} viewpoints")   
    breakpoint()
    for i,viewpoint in tqdm(enumerate(cameras)):
        viewpoint=viewpoint
        viewpoint.world_view_transform = viewpoint.world_view_transform.cuda()
        # viewpoint.projection_matrix = viewpoint.projection_matrix.cuda()
        viewpoint.full_proj_transform = viewpoint.full_proj_transform.cuda()
        viewpoint.camera_center = viewpoint.camera_center.cuda()

        # tanfovx = math.tan(viewpoint.fovX * 0.5)
        # threshold = (2 * (tau + 0.5)) * tanfovx / (0.5 * viewpoint.image_width)

        # Render 
        image = torch.clamp(render(
            viewpoint, 
            scene.gaussians, 
            pipe, 
            torch.tensor([0.0, 0.0, 0.0], dtype=torch.float32, device="cuda"), 
            indices= None,
            use_trained_exp=args.train_test_exp
            )["render"], 0.0, 1.0)


        if args.camera_novel_paths is not None:
            save_path = Path(render_path) / 'rgb' / f"{i:05d}.png"
            os.makedirs(save_path.parent, exist_ok=True)
            torchvision.utils.save_image(image, save_path)
        else:
            save_path = Path(render_path) / 'rgb' / Path(viewpoint.image_name).with_suffix('.png')
            os.makedirs(save_path.parent, exist_ok=True)
            torchvision.utils.save_image(image, save_path)
        
    torch.cuda.empty_cache()


if __name__ == "__main__":
    # Set up command line argument parser
    parser = ArgumentParser(description="Rendering script parameters")
    lp = ModelParams(parser)
    op = OptimizationParams(parser)
    pp = PipelineParams(parser)
    parser.add_argument('--out_dir', type=str, default="")
    parser.add_argument("--taus", nargs="+", type=float, default=[0.0, 10.0])
    parser.add_argument("--dataset_type", default='colmap')
    parser.add_argument("--camera_novel_paths", default=None) 
    args = parser.parse_args(sys.argv[1:])
    
    print("Rendering " + args.model_path)

    dataset, pipe = lp.extract(args), pp.extract(args)
    gaussians = GaussianModel(dataset.sh_degree)
    gaussians.active_sh_degree = dataset.sh_degree
    
    scene  = Scene(dataset, gaussians, load_iteration=-1, create_from_hier=False, kargs=args)
    render_novel_view(args, scene, pipe, os.path.join(args.out_dir, f"render_novel"), 0.0, args.eval)
