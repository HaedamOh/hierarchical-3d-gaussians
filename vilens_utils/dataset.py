import re
from pathlib import Path 
import yaml
import numpy as np 
import open3d as o3d
import evo 
from evo.tools.file_interface import csv_read_matrix
from evo.core.trajectory import PosePath3D
from .sensor import Sensor
from .file_interfaces.vilens_slam import VilensSlamTrajReader
from .file_interfaces.timestamp import TimeStamp
# from sensor import Sensor
# from file_interfaces.vilens_slam import VilensSlamTrajReader
# from file_interfaces.timestamp import TimeStamp


'''
Edit Nerf_data_pipeline/dataset/dataset.py

'''

def read_pose_csv_file(file_path):
    """
    Read VILENS SLAM trajectory csv file 
    @return: PosePath3D from evo
    """
    raw_mat = csv_read_matrix(file_path, delim=",", comment_str="#")
    if not raw_mat:
        raise ValueError()
    sec = raw_mat[:, 0]
    nsec = raw_mat[:, 1]
    timestamps = float(sec) + float(nsec) * 1e-9
    mat = np.array(raw_mat).astype(float)
    xyz = mat[:, 3:6]
    quat_xyzw = mat[:, 6:10]
    quat_wxyz = np.roll(quat_xyzw, 1, axis=1)  # xyzw -> wxyz
    return evo.core.trajectory.PoseTrajectory3D(xyz, quat_wxyz, timestamps=timestamps)


class VilensSlamOutputHandler:
    def __init__(
        self, 
        slam_traj_path: Path,
        image_traj_path: Path, 
        slam_clouds_folder_path: Path,
        downsample_to: int = -1
    ):
        self.slam_traj_path = Path(slam_traj_path)
        self.image_traj_path = Path(image_traj_path)
        self.slam_traj = self.get_traj(self.slam_traj_path)
        self.image_traj = self.get_traj(self.image_traj_path) 
        self.slam_num_poses = self.slam_traj.num_poses
        self.image_num_poses = self.image_traj.num_poses
        print(f"SLAM poses: {self.slam_num_poses}, Synced image poses: {self.image_num_poses}")
        self.slam_clouds_folder_path = Path(slam_clouds_folder_path)

        self.check_clouds_with_poses()
        if downsample_to > 0:
            print(f"Downsampling to {downsample_to} poses")
            self.slam_traj.downsample(downsample_to)
        
    def get_traj(self, traj_path: Path):
        return VilensSlamTrajReader(traj_path).read_file()

    def get_cloud_path(self, pose_idx: int):
        timestamp = TimeStamp(t_float128=self.slam_traj.timestamps[pose_idx])
        timestamp_str = timestamp.t_string
        timestamp_str = timestamp_str.replace(".", "_")  # vilens slam output convention
        cloud_file = self.slam_clouds_folder_path / f"cloud_{timestamp_str}.pcd"
        return cloud_file

    def get_cloud_pcd(self, pose_idx: int):
        cloud_file = self.get_cloud_path(pose_idx)
        return o3d.io.read_point_cloud(str(cloud_file))

    def get_slam_pose(self, pose_idx: int):
        return self.slam_traj.poses_se3[pose_idx]

    def get_image_pose(self, pose_idx: int):
        return self.image_traj.poses_se3[pose_idx]

    def check_clouds_with_poses(self):
        cloud_files = list(self.slam_clouds_folder_path.glob("*.pcd"))
        assert all([self.get_cloud_path(pose_idx) in cloud_files for pose_idx in range(self.slam_traj.num_poses)])


class FrontierDataset:
    '''
    ROS2 VILENS outputs folder 
    '''
    def __init__(
        self,
        slam_output_folder_path: Path,
        sensor_config_yaml_path: Path,
        slam_poses_csv: str = "slam_poses.csv",
        image_poses_csv: str = "image_poses.csv",
        slam_pose_graph_slam: str = "slam_pose_graph.slam",
        slam_clouds_folder: str = "slam_clouds",
        image_folder: str = "images_rectified",
        slam_pose_downsample_to: int = -1,  # downsample poses. -1 means no downsampling
    ):
        self.slam_poses_csv_path = Path(slam_output_folder_path) / slam_poses_csv
        self.image_poses_csv_path = Path(slam_output_folder_path) / image_poses_csv
        self.slam_clouds_folder_path = Path(slam_output_folder_path) / slam_clouds_folder
        self.image_folder_path = Path(slam_output_folder_path) / image_folder
        
        self.vilens_slam_handler = VilensSlamOutputHandler(
            self.slam_poses_csv_path, 
            self.image_poses_csv_path, 
            self.slam_clouds_folder_path,
            slam_pose_downsample_to
        )
        
        self.sensor = Sensor.load_config(Path(sensor_config_yaml_path))        
        self.load_synced_images()

    def get_all_camera_poses(self, return_dict=False, sync_with_images=False, visualize=False,timestamp_ranges=None):
        """
        Retrieves camera poses, optionally filtered by timestamps.
        
        Args:
            return_dict (bool): If True, returns a dictionary with camera labels as keys.
            sync_with_images (bool): If True, assumes camera images are synced with poses.
            visualize (bool): If True, visualizes camera poses as coordinate frames.
            timestamp1 (float): Start timestamp for filtering poses. Defaults to None.
            timestamp2 (float): End timestamp for filtering poses. Defaults to None.
            
        Returns:
            tuple: (camera_poses, camera_paths), either as dicts or lists depending on `return_dict`.
        """
        cam_poses_vis = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        camera_poses = {} if return_dict else []
        camera_paths = {} if return_dict else []
        camera_timestamps = {} if return_dict else []
        
        for camera in self.sensor.cameras:
            if return_dict:
                camera_poses[camera.label] = []
                camera_paths[camera.label] = []
                camera_timestamps[camera.label] = []

            for i in range(self.vilens_slam_handler.image_num_poses):
                timestamp = TimeStamp(t_float128=self.vilens_slam_handler.image_traj.timestamps[i]).t_float128
                
                # Filter by multiple timestamp ranges
                if timestamp_ranges:
                    if not any(start <= timestamp <= end for start, end in timestamp_ranges):
                        continue
                T_WB = self.vilens_slam_handler.get_image_pose(i)
                T_WC = T_WB @ self.sensor.tf.get_transform(camera.label, "base")

                if return_dict:
                    camera_poses[camera.label].append(T_WC)
                    camera_paths[camera.label].append(self.image_paths[camera.label][timestamp])
                    camera_timestamps[camera.label].append(timestamp)
                else:
                    camera_poses.append(T_WC)
                    camera_paths.append(self.image_paths[camera.label][timestamp])
                    camera_timestamps.append(timestamp)
                    
                if visualize:
                    cam_poses_vis += o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).transform(T_WC)

        if not return_dict:
            camera_poses = np.array(camera_poses)
            camera_poses = PosePath3D(poses_se3=camera_poses)

        if visualize:
            o3d.visualization.draw_geometries([cam_poses_vis])

        return camera_poses, camera_paths, camera_timestamps

    def load_synced_images(self, image_ext=".jpg"):
        '''
        Returns:
            self.image_paths: dict, {camera_label: {timestamp: Path}} 
        '''
        self.image_paths = {}
        for camera in self.sensor.cameras:
            image_timestamps = []
            image_paths = {}    # timestamp: Path 
            cam_topic_folder = camera.label
            image_folder_path = self.image_folder_path / cam_topic_folder 
            for it in sorted(list(image_folder_path.glob(f"*{image_ext}"))):
                ret = re.findall(r"\d+", it.name)
                timestamp = str(".".join(ret))
                timestamp = TimeStamp(t_string=timestamp).t_float128
                image_timestamps.append(timestamp)
                image_paths[timestamp] = it
            assert len(image_paths) > 0, "No images are found"
            self.image_paths[camera.label] = image_paths

    def get_map_cloud(self, step=1, downsample_res=0.1, save_path=None, visualize=False):
        map_cloud = o3d.geometry.PointCloud()
        base_poses = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        for i in range(0, self.vilens_slam_handler.slam_num_poses, step):
            # read cloud
            cloud = self.vilens_slam_handler.get_cloud_pcd(i)
            T_WB = self.vilens_slam_handler.get_slam_pose(i)
            cloud.transform(T_WB)
            map_cloud += cloud
            # add pose 
            base_poses += o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).transform(T_WB)
        map_cloud = map_cloud.voxel_down_sample(downsample_res)
        
        if save_path is not None:
            o3d.io.write_point_cloud(save_path, map_cloud)
        if visualize:
            o3d.visualization.draw_geometries([map_cloud, base_poses])
        
        return map_cloud
    
    def visualize_camera_poses(self):
        map_cloud = o3d.geometry.PointCloud()
        # Base frame ( in Robotics )
        base_poses = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        for i in range(self.vilens_slam_handler.slam_num_poses):
            T_WB = self.vilens_slam_handler.get_slam_pose(i)
            base_poses += o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).transform(T_WB)
            cloud = self.vilens_slam_handler.get_cloud_pcd(i)
            cloud.transform(T_WB)
            map_cloud += cloud
        
        # Camera frames (in Computer Vision)
        image_poses = o3d.geometry.TriangleMesh.create_coordinate_frame(size=3)
        for camera in self.sensor.cameras:
            for i in range(self.vilens_slam_handler.image_num_poses):
                T_WB = self.vilens_slam_handler.get_image_pose(i)
                T_WC = self.sensor.tf.get_transform(camera.label, "base")
                T_WC = T_WB @ T_WC
                image_poses += o3d.geometry.TriangleMesh.create_coordinate_frame(size=3).transform(T_WC)
        print("Visualizing camera frames")
        o3d.visualization.draw_geometries([map_cloud,  image_poses])
        

if __name__ == "__main__":

    '''
    Example: Frontier dataset reads VILENS SLAM outputs and load Sensor internal class
    '''

    dataset = FrontierDataset(
        slam_output_folder_path=Path("/home/haedam/vilens_slam_data/2024-05-20-Bodleian-02/vilens_map/2024-12-10_15-23-40_rec003"),
        sensor_config_yaml_path=Path("/home/haedam/git/runtime_drs/config/frn019/sensors_2024_10_29/sensors_frn019.yaml"),
        slam_poses_csv="slam_poses.csv",
        image_poses_csv="image_poses.csv",
        slam_pose_graph_slam="slam_pose_graph.slam",
        slam_clouds_folder="slam_clouds",
        image_folder="images_rectified",
        slam_pose_downsample_to=-1
    )

    camera_poses, camera_paths = dataset.get_all_camera_poses(return_dict=True, sync_with_images=False, visualize=True) 
    print(camera_poses.keys())
    cam_intrinsics = dataset.sensor.cameras[0].rect_intrinsics 
    print(cam_intrinsics)

    # debug 
    dataset.visualize_camera_poses()
    
    