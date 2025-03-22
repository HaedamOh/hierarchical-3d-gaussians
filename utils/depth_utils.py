import numpy as np
import matplotlib.pyplot as plt
import torch
import cv2
import open3d as o3d
from typing import Optional
import struct
from pathlib import Path

def visualize_l1_depth_error(depth1, depth2, depth_mask, save_path, error_range=10):
    # Assume inputs are torch tensors
    depth1 = depth1.squeeze(0).detach().cpu().numpy()
    depth2 = depth2.squeeze(0).detach().cpu().numpy()
    depth_mask = depth_mask.squeeze(0).cpu().detach().numpy().astype(bool)
    
    depth_diff = np.full_like(depth1, np.nan)
    
    depth_diff[depth_mask] = np.abs(depth1[depth_mask] - depth2[depth_mask])
    depth_diff_clipped = np.clip(depth_diff, 0, error_range)  # Cap values within range
    print(f"Min Error: {np.nanmin(depth_diff)}, Max Error: {np.nanmax(depth_diff)}")
    percentile_80_error = np.nanpercentile(depth_diff, 80)
    print(f"80th Percentile Error: {percentile_80_error}")
    norm_depth_diff = depth_diff_clipped / error_range  # Normalize in [0,1]    
    colormap = plt.get_cmap('jet')
    color_mapped_image = colormap(norm_depth_diff)[:, :, :3]  # RGB only

    # Convert NaN areas to grayscale 
    color_mapped_image[np.isnan(depth_diff)] = [0.5, 0.5, 0.5]  # Gray for invalid regions    
    color_mapped_image = (color_mapped_image * 255).astype(np.uint8)
    
    # Save as image
    cv2.imwrite(save_path, cv2.cvtColor(color_mapped_image, cv2.COLOR_RGB2BGR))
    print(f'L1 Depth Error Map saved to {save_path}')
    
def convert_invdepth_to_depth(invdepthmap):
    ''' Torch invdepthmap'''
    if not isinstance(invdepthmap, torch.Tensor):
        invdepthmap = torch.tensor(invdepthmap, dtype=torch.float32)
    depth = torch.zeros_like(invdepthmap)

    valid_invdepth = torch.logical_and(invdepthmap > 0, invdepthmap < 255)
    depth[valid_invdepth] = 1.0 / invdepthmap[valid_invdepth]

    depth[invdepthmap >= 255] = 0.0

    depth[invdepthmap == 0] = 255.0
    depth[~valid_invdepth] = torch.nan
    return depth

def convert_depth_to_invdepth(depth):
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=torch.float32)
    invdepth = torch.zeros_like(depth)

    valid_depth = torch.logical_and(depth > 0, depth < 255)
    invdepth[valid_depth] = 1.0 / depth[valid_depth]

    invdepth[depth >= 255] = 0.0

    invdepth[depth == 0] = 255.0
    invdepth[~valid_depth] = torch.nan
    return invdepth

def get_overlay(camera_image, depth_image, cmap="hsv", circle_radius=2, circle_thickness=2):
    if isinstance(camera_image, torch.Tensor):
        camera_image = camera_image.detach().cpu().numpy().transpose(1, 2, 0)  # (C, H, W) -> (H, W, C)
    if isinstance(depth_image, torch.Tensor):
        depth_image = depth_image.squeeze(0).detach().cpu().numpy()  # (1, H, W) -> (H, W)
    
    # Normalize depth image
    depth_mask = depth_image > 0
    if depth_image.max() == 0:
        print("WARNING: Depth image is all zeros")
        return camera_image
    # Normalize and apply colormap
    depth_norm = depth_image / np.max(depth_image)
    cmap = plt.cm.get_cmap(cmap)
    depth_image_cmap = cmap(depth_norm)[:, :, :3]  # Remove the alpha channel
    depth_image_cmap = (depth_image_cmap * 255).astype(np.uint8)
    
    # Apply depth mask
    depth_image_cmap[~depth_mask] = 0
    depth_image_cmap = cv2.cvtColor(depth_image_cmap, cv2.COLOR_RGB2BGR)

    # Create the overlay
    overlay = camera_image.copy()
    for x, y in zip(*np.where(depth_mask)):
        # Overlay circles on the depth image where depth exists
        cv2.circle(
            overlay,
            (y, x),  # (y, x) for OpenCV's (x, y) format
            circle_radius,
            depth_image_cmap[x, y].tolist(),
            circle_thickness,
        )

    # Finally, overlay the depth colormap on top of the camera image
    overlay[depth_mask] = depth_image_cmap[depth_mask]
    return overlay



def convert_uint8_depth_to_invdepth(depth):
    """
    Convert a uint8 depth map (0-255) to an inverse depth map (also 0-255).

    Args:
        depth (torch.Tensor or np.ndarray): Depth map in uint8 format.

    Returns:
        torch.Tensor: Inverse depth map as uint8.
    """
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=torch.uint8)

    # Compute inverse depth (flip depth values)
    inv_depth = 255 - depth

    return inv_depth.to(torch.uint8)

def convert_uint16_depth_to_invdepth(depth):
    """
    Convert a 16-bit uint16 depth map (0-65535) to an inverse depth map (also 0-65535).

    Args:
        depth (torch.Tensor or np.ndarray): Depth map in uint16 format.

    Returns:
        torch.Tensor: Inverse depth map as uint16.
    """
    if not isinstance(depth, torch.Tensor):
        depth = torch.tensor(depth, dtype=torch.uint16)

    # Compute inverse depth (flip depth values)
    inv_depth = 65535 - depth

    return inv_depth.to(torch.uint16)




def encode_points_as_depthmap(
    points_on_img: np.ndarray,
    points_in_3d: np.ndarray,
    h: int,
    w: int,
    is_euclidean: bool,
    depth_encode_factor: float,
    point_size: float = 1.0,
    normals: Optional[np.ndarray] = None,
    K: Optional[np.ndarray] = None,
    D: Optional[np.ndarray] = None,
) -> np.ndarray:
    """
    Encode a set of 3D points as a depth map. It is assumed to be called after `project_pcd_on_image`.

    Args:
        points_on_img (np.ndarray): A numpy array of shape (N, 2) representing the (x, y) coordinates of the projected 3D
            points on the image plane.
        points_in_3d (np.ndarray): A numpy array of shape (N, 3) representing the corresponding 3D coordinates of the
            projected points in the input point cloud.
        h (int): The height of the output depth map in pixels.
        w (int): The width of the output depth map in pixels.
        is_euclidean (bool): A flag indicating whether to use the Euclidean distance between the points and the camera
            as the depth value. If False, the z coordinate of each point is used instead.
        depth_encode_factor (float): A scaling factor applied to the depth values to convert them to 16-bit unsigned
            integers. The maximum depth value that can be represented is (2^16 - 1) / depth_encode_factor.

    Returns:
        np.ndarray: A 2D numpy array of shape (h, w) representing the encoded depth map. Each pixel value is a 16-bit
            unsigned integer representing the depth value of the corresponding point in the input point cloud. Pixels
            without any corresponding point are set to 0.
    """
    # Extract depth
    if is_euclidean:
        # Depth is L2 distance between point and camera
        depth = np.linalg.norm(points_in_3d, axis=1)
    else:
        # Depth is z value
        depth = points_in_3d[:, 2]
    # Later depth is saved as 16 bit png (0 - 65,535)
    # Remove outside 16 bit range
    z_mask = (depth * depth_encode_factor) < np.iinfo(np.uint16).max

    if not z_mask.all():
        print("[warn] Depth values are too large to be encoded as 16-bit unsigned integers.")

    # Extract only valid value
    valid_points_on_img = points_on_img[z_mask]
    valid_depth = depth[z_mask]
    # valid_points_in_3d = points_in_3d[z_mask]

    depthmap = np.zeros((h, w), dtype=np.uint16)
    u, v = valid_points_on_img.transpose().astype(int)
    sorted_indices = np.argsort(valid_depth)[::-1]
    u = u[sorted_indices]
    v = v[sorted_indices]
    valid_depth = valid_depth[sorted_indices]
    z = (depth_encode_factor * valid_depth).astype(np.uint16)

    # size-aware rendering
    # point_radii = point_size / valid_depth
    # unique_x, unique_y, unique_z = size_aware_rendering(u, v, z, point_radii, h, w)
    # depthmap[unique_y, unique_x] = unique_z

    depthmap[v, u] = z



def decode_points_from_depthmap(
    depth: np.ndarray,
    K: np.ndarray,
    D: np.ndarray,
    is_euclidean: bool,
    depth_encode_factor: float,
    image: Optional[np.ndarray] = None,
    camera_model: str = "OPENCV",
    is_inverse_depth: bool = False,
) -> o3d.geometry.PointCloud:

    # png uint16 encoding value -> m
    depth = depth.astype(float) / depth_encode_factor
    h, w = depth.shape

    # Valid depth mask
    mask = depth > 0
    valid_depth = depth[mask] # metric depth in meters
    
    breakpoint()
    # Get coordinates of valid depth
    grid_x, grid_y = np.meshgrid(np.arange(w), np.arange(h))
    grid_x = grid_x[mask]
    grid_y = grid_y[mask]

    # Project points on image into 3d world coordinates
    points_on_img = np.stack((grid_x, grid_y), axis=1)
    # Adjust shape and type for cv2.fisheye.undistortPoints
    points_on_img = points_on_img[np.newaxis].astype(float)
    # Project points on image onto z = 1 plane
    if camera_model == "OPENCV_FISHEYE":
        points_in_3d = cv2.fisheye.undistortPoints(points_on_img, K, D, P=np.eye(3))
    elif camera_model == "OPENCV":
        points_in_3d = cv2.undistortPoints(points_on_img, K, D, P=np.eye(3)) #(n,1,2)
    else:
        raise ValueError(f"Unknown camera model: {camera_model}")
    if points_in_3d is None:
        return o3d.geometry.PointCloud()
    points_in_3d = points_in_3d.squeeze() # (n,2)
    # Add z value which is one
    # TODO: how to handle one point properly?
    if points_in_3d.ndim == 1:
        points_in_3d = points_in_3d[np.newaxis] 

    points_in_3d = np.concatenate((points_in_3d, np.ones((len(points_in_3d), 1))), axis=1) # (n,3) in homogeneous coordinates
    if is_euclidean:
        # Normalize since depth is range data (L2 distance between camera and point)
        points_in_3d /= np.linalg.norm(points_in_3d, axis=1, keepdims=True)
    # Multiple by depth
    if is_inverse_depth:
        points_in_3d = (1 / valid_depth[:, np.newaxis]) * points_in_3d
    else:
        points_in_3d = valid_depth[:, np.newaxis] * points_in_3d
    
    # Convert into point cloud
    if image is not None:
        colors = image[mask]
    else:
        colors = None
        
    def get_pcd(points, colors=None):
        """Get open3d pointcloud from points and colors"""
        pcd = o3d.geometry.PointCloud()
        pcd.points = o3d.utility.Vector3dVector(points)
        if colors is not None:
            assert colors.max() <= 1.0 and colors.min() >= 0.0
            pcd.colors = o3d.utility.Vector3dVector(colors)
        return pcd
    pcd = get_pcd(points=points_in_3d, colors=colors)

    return pcd

def encode_depthmap_to_colormap(depthmap,depth_encoded_factor=256, cmap="jet", save_path='colormap_depth.png'):

    # Metric depth 
    depthmap = depthmap / depth_encoded_factor 
    valid_mask = depthmap > 0
    print('Number of valid mask:',  np.sum(valid_mask), 'out of', depthmap.size)
    
    # Filter
    y_idxs, x_idxs = np.where(valid_mask)  
    valid_depths = depthmap[valid_mask]    

    min_depth = np.min(valid_depths)
    max_depth = np.max(valid_depths)

    # Use scatter plot for dots instead of colormap
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_idxs, y_idxs, c=valid_depths, cmap=cmap, vmin=min_depth, vmax=max_depth, s=1)

    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (Absolute Scale)")
    cbar.set_ticks([min_depth, max_depth])
    cbar.set_ticklabels([f"{min_depth}", f"{max_depth}"])

    plt.gca().invert_yaxis()  # Flip Y-axis to match image convention
    plt.title("Depth Map Visualization")
    plt.axis("off")
    plt.savefig(save_path)

def encode_invdepthmap_to_colormap(inverse_depth_path,depth_params, depth_encoded_factor=256, cmap="jet",save_path = None):
    import json 

    image_key = Path(inverse_depth_path).parent.stem + "/" + Path(inverse_depth_path).stem
    depth_params = depth_params[image_key]
    
    invdepthmap = cv2.imread(inverse_depth_path, cv2.IMREAD_ANYDEPTH)
    invdepthmap = invdepthmap / float(2**16) # 0 - 1
    valid_mask = invdepthmap > 0 
    invdepthmapScaled = invdepthmap * depth_params["scale"] + depth_params["offset"]
        
    valid_depth = np.zeros_like(invdepthmap) 
    valid_depth[valid_mask] = 1 / invdepthmapScaled[valid_mask]

    min_depth = np.min(valid_depth)
    max_depth = np.max(valid_depth)
    # Filter
    y_idxs, x_idxs = np.where(valid_mask)
    # Use scatter plot for dots instead of colormap
    plt.figure(figsize=(8, 6))
    scatter = plt.scatter(x_idxs, y_idxs, c=valid_depth[valid_mask], cmap=cmap, vmin=min_depth, vmax=max_depth, s=1)
    
    cbar = plt.colorbar(scatter, fraction=0.046, pad=0.04)
    cbar.set_label("Depth (Absolute Scale)")
    cbar.set_ticks([min_depth, max_depth])
    cbar.set_ticklabels([f"{min_depth}", f"{max_depth}"])
    
    plt.gca().invert_yaxis()  # Flip Y-axis to match image convention
    plt.title("Inverse Depth Map Visualization")
    plt.axis("off")
    plt.savefig(save_path)


if __name__ == '__main__':
    ''' Debug rendered depth from 3DGS'''

    ''' Test NDP depth to invdepth conversion '''
    
    # InvDepth 
    ndp_inverse_depth_path = '/home/shared/oxford_spires/processed_ndp/2024-03-13-roq_2/processed/depths_z_accum_3_rect_inverse/alphasense_driver_ros_cam0_debayered_image_compressed/1710338516.736862820.png'
    ndp_inverse_depth = cv2.imread(ndp_inverse_depth_path, cv2.IMREAD_ANYDEPTH) 

    # Depth 
    ndp_depth_path = '/home/shared/nerfstudio/oxford_spires/2024-03-13-maths_2/depths_z_accum_3_rect/alphasense_driver_ros_cam0_debayered_image_compressed/1710338516.736862820.png'
    ndp_depth = cv2.imread(ndp_depth_path, cv2.IMREAD_ANYDEPTH)
    
    # Image 
    ndp_image_path ='/home/shared/nerfstudio/oxford_spires/2024-03-13-maths_2/images_rectified/alphasense_driver_ros_cam0_debayered_image_compressed/1710338516.736862820.jpg'
    ndp_image = cv2.imread(ndp_image_path) 
    ndp_image = cv2.cvtColor(ndp_image, cv2.COLOR_BGR2RGB) / 255.0
    
    # alphasense_cam0 (cam front) 
    K = np.array([[567.1327295190782, 0,   700.119268458472       ],
                [0, 567.4134238587335, 573.8848164092475      ],
                [  0. ,          0.   ,        1. ,       ]])
    depth_param = {"alphasense_driver_ros_cam0_debayered_image_compressed/1710338516.736862820": {
        "scale": 0.49034963065295695,
        "offset": 0.018359339928276177
    }}
    # pcd = decode_points_from_depthmap(ndp_inverse_depth,
    #                                   K=K, D=np.zeros(4),
    #                                   is_euclidean=False,
    #                                   depth_encode_factor=256.0,
    #                                   image=ndp_image,
    #                                   camera_model="OPENCV",
    #                                   is_inverse_depth=True)
    # o3d.io.write_point_cloud('gt_test.ply', pcd)
    
    
    # encode_depthmap_to_colormap(ndp_depth, cmap="jet", save_path='colormap_depth.png') # Works ! 
    
    # depth_params_json = '/home/shared/oxford_spires/processed_ndp/2024-03-13-roq_2/processed/depth_params.json'
    encode_invdepthmap_to_colormap(ndp_inverse_depth_path,depth_param, cmap="jet", save_path='colormap_invdepth.png')
    
    ''' Test INRIA invdepth to depth conversion  '''
    
    ''' {
    1: Camera(
        id=1, model='PINHOLE', width=1028, height=686,
        params=array([479.03628728, 477.6571608 , 514. , 343.])
    ),
    2: Camera(
        id=2, model='PINHOLE', width=1022, height=690,
        params=array([485.05912252, 483.88831544, 511. , 345.])
    ),
    3: Camera(
        id=3, model='PINHOLE', width=1020, height=685,
        params=array([481.16391966, 483.02459008, 510. , 342.5])
    ),
    4: Camera(
        id=4, model='PINHOLE', width=1025, height=687,
        params=array([481.74661467, 485.8119994 , 512.5, 343.5])
    ),
    5: Camera(
        id=5, model='PINHOLE', width=1024, height=690,
        params=array([484.40787587, 483.26718308, 512. , 345.])
    ),
    6: Camera(
        id=6, model='PINHOLE', width=1027, height=687,
        params=array([481.30179853, 480.85607954, 513.5, 343.5])
    )
    }
    '''
    # irnia_inv_z_depth = './example_dataset/camera_calibration/rectified/depths/cam5/pass1_1036.png'
    # inria_image = './example_dataset/camera_calibration/rectified/images/cam5/pass1_1036.jpg'
    
    # inria_image = cv2.imread(inria_image)
    # inria_image = cv2.cvtColor(inria_image, cv2.COLOR_BGR2RGB) / 255.0

    # irnia_inv_z_depth = cv2.imread(irnia_inv_z_depth, cv2.IMREAD_ANYDEPTH).astype(np.uint16) # 16 bit png (0 - 65,535) 
    # inria_inv_z_depth_metric = irnia_inv_z_depth / 256.0
    
    # valid_depth_mask = inria_inv_z_depth_metric > 0
    
    # inria_z_depth_metric = np.zeros_like(inria_inv_z_depth_metric)
    # inria_z_depth_metric[valid_depth_mask] = 1 / inria_inv_z_depth_metric[valid_depth_mask]
    
    
    # inria_z_depth = (inria_z_depth_metric * 256).astype(np.uint16) 
    
    

    
    # K = np.array([[484.40787587,     0,         512       ],
    #               [0,           483.26718308,    345.       ],
    #               [0.,          0.,              1. ,      ]])
    # # pcd = inverse_depth_to_pointcloud(inria_image, inria_z_depth, K)
    # pcd = decode_points_from_depthmap(inria_z_depth, K=K, D=np.zeros(4), is_euclidean=False, depth_encode_factor=256.0, image=inria_image, camera_model="OPENCV")
    # o3d.io.write_point_cloud('inria_gt_test.ply', pcd)
    # print('pcd points:', len(pcd.points))
    ''' Test INRIA depth to invdepth conversion '''
