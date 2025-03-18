import open3d as o3d
import os
import numpy as np
import json 

def voxel_downsample(pcd_path, voxel_size=0.5):
    point_cloud = o3d.io.read_point_cloud(pcd_path)
    downsampled_point_cloud = point_cloud.voxel_down_sample(voxel_size)

    downsampled_path = pcd_path.replace(".pcd", "_downsampled_" + str(voxel_size) + ".pcd")

    o3d.io.write_point_cloud(downsampled_path, downsampled_point_cloud)
    print("Number of points before downsample: ", len(point_cloud.points))
    print("Saved downsampled point cloud at: ", downsampled_path)
    return downsampled_path


def random_downsample(pcd_path, final_num_points=200000):
    point_cloud = o3d.io.read_point_cloud(pcd_path)
    current_num_points = len(point_cloud.points)
    if final_num_points >= current_num_points:
        print(f"Final number of points ({final_num_points}) is greater than or equal to the current number of points ({current_num_points}). No downsampling needed.")
        return pcd_path
    downscale = current_num_points // final_num_points
    downsampled_point_cloud = point_cloud.uniform_down_sample(downscale)
    downsampled_path = pcd_path.replace(".pcd", f"_downsampled_{final_num_points}.pcd")
    o3d.io.write_point_cloud(downsampled_path, downsampled_point_cloud)
    print("Number of points before downsample: ", current_num_points)
    print("Number of points after downsample: ", len(downsampled_point_cloud.points))
    return downsampled_path


def visualize_pcd(pcd_path):
    pcd = o3d.io.read_point_cloud(pcd_path)
    # Step 2: Verify if normals are loaded
    if not pcd.has_normals():
        print("No normals found in the PCD file.")
    else:
        print("Normals are loaded successfully.")

    # Step 3: Visualize the point cloud with normals
    # Set up visualizer with point cloud and normals
    print('Colors:',np.asarray(pcd.colors))
    
    o3d.visualization.draw_geometries(
        [pcd],
        zoom=0.8,
        front=[0.0, 0.0, -1.0],
        lookat=[0.0, 0.0, 0.0],
        up=[0.0, -1.0, 0.0],
        point_show_normal=True  # Enable normals visualization
    )


def combine_individual_clouds(pcd_root_folder):
    pcd_files = []
    for root, _, files in os.walk(pcd_root_folder):
        pcd_files.extend([os.path.join(root, f) for f in files if f.endswith(".pcd")])    
    if not pcd_files:
        raise FileNotFoundError(f"No .pcd files found in {pcd_root_folder}")
    combined_pcd = o3d.geometry.PointCloud()
    for pcd_file in pcd_files:
        point_cloud = o3d.io.read_point_cloud(pcd_file)
        combined_pcd += point_cloud
    combined_pcd = combined_pcd.uniform_down_sample(10)
    root_folder = os.path.join(pcd_root_folder, "..")
    output_path = os.path.join(root_folder, "combined_colorized_cloud.pcd")
    o3d.io.write_point_cloud(output_path, combined_pcd)
    print("Number of points ", len(combined_pcd.points))
    print("saved combined point cloud at: ", output_path)
    return output_path



import numpy as np
import json
from plyfile import PlyData, PlyElement
from scipy.spatial.transform import Rotation as R

def transform_gaussian_ply(ply_path, evo_transforms_json):
    # Load transformation parameters
    with open(evo_transforms_json, "r") as f:
        data = json.load(f)
        scale = data['scale']
        rotation = np.array(data['rotation']).reshape(3, 3)
        translation = np.array(data['translation'])

        # Construct 4x4 transformation matrix
        transformation = np.eye(4)
        transformation[:3, :3] = rotation * scale
        transformation[:3, 3] = translation

    # Read PLY data
    ply_data = PlyData.read(ply_path)
    vertex_data = ply_data['vertex'].data  # Access structured NumPy array
    attribute_names = vertex_data.dtype.names  # Get all field names
    
    # Transform positions
    xyz = np.vstack((vertex_data['x'], vertex_data['y'], vertex_data['z'])).T
    xyz_h = np.hstack((xyz, np.ones((xyz.shape[0], 1))))
    xyz_transformed = (transformation @ xyz_h.T).T[:, :3]

    # Apply scale transformation to 'scale_0', 'scale_1', and 'scale_2'
    xyz_scale_transformed = scale * np.vstack((vertex_data['scale_0'], vertex_data['scale_1'], vertex_data['scale_2'])).T  # (n, 3)

    # Apply rotation transformation to 'rot_0', 'rot_1', 'rot_2', 'rot_3' (quaternion rotation)
    quats = np.vstack((vertex_data['rot_0'], vertex_data['rot_1'], vertex_data['rot_2'], vertex_data['rot_3'])).T
    rotations = R.from_quat(quats)  # Convert quaternion to rotation matrix
    rotated_xyz = rotation @ rotations.as_matrix() @ rotation.T  # Apply additional rotation
    rotated_quats = R.from_matrix(rotated_xyz).as_quat()  # Convert back to quaternion

    # Create new vertices with updated properties
    new_vertices = np.empty(len(vertex_data), dtype=vertex_data.dtype)

    for name in attribute_names:
        if name == 'x':
            new_vertices[name] = xyz_transformed[:, 0]
        elif name == 'y':
            new_vertices[name] = xyz_transformed[:, 1]
        elif name == 'z':
            new_vertices[name] = xyz_transformed[:, 2]
        elif name == 'scale_0':  # Apply scale transformation
            new_vertices[name] = xyz_scale_transformed[:, 0]
        elif name == 'scale_1':  # Apply scale transformation
            new_vertices[name] = xyz_scale_transformed[:, 1]
        elif name == 'scale_2':  # Apply scale transformation
            new_vertices[name] = xyz_scale_transformed[:, 2]
        elif name == 'rot_0':  # Apply rotation transformation to 'rot_0' component
            new_vertices[name] = rotated_quats[:, 0]
        elif name == 'rot_1':  # Apply rotation transformation to 'rot_1' component
            new_vertices[name] = rotated_quats[:, 1]
        elif name == 'rot_2':  # Apply rotation transformation to 'rot_2' component
            new_vertices[name] = rotated_quats[:, 2]
        elif name == 'rot_3':  # Apply rotation transformation to 'rot_3' component
            new_vertices[name] = rotated_quats[:, 3]
        else:
            new_vertices[name] = vertex_data[name]  # Copy all other attributes

    # Create a new PLY element and save the transformed data
    new_ply_element = PlyElement.describe(new_vertices, 'vertex')
    output_path = ply_path.replace(".ply", "_transformed.ply")
    PlyData([new_ply_element]).write(output_path)

    print(f"Transformed PLY saved to {output_path}")

if __name__ == "__main__":
    '''
    Downsample input pointcloud for training
    '''

    # voxel_downsample("/home/haedam/vilens_slam_data/2024-05-20-Bodleian-02/vilens_map/2024-12-23_17-19-04_rec009/slam_combined_cloud_small.pcd")
    # random_downsample("data/2024-03-13-maths_2/gaussian_splatting/combined_colorized_cloud.pcd")
    # visualize_pcd("/home/haedam/vilens_slam_data/2024-05-20-Bodleian-02/vilens_map/2024-12-23_17-19-04_rec009/slam_combined_cloud_small.pcd")

    # combine_individual_clouds('/home/haedam/git/3DGS_mapping/data/2024-05-20-bod_2/depths_euc_accum_3_cloud')

    transform_gaussian_ply(ply_path='/home/shared/frontier_data/fnt_802/2024-12-13-10-24-rec001/processed/output_colmap/output/aligned/point_cloud/iteration_30000/point_cloud.ply',
                    evo_transforms_json='/home/shared/frontier_data/fnt_802/2024-12-13-10-24-rec001/processed/output_colmap/camera_calibration/aligned/evo_align_results.json')