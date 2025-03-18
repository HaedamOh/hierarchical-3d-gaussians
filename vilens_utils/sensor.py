import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import List, Dict

import numpy as np
import yaml
from evo.core.trajectory import xyz_quat_wxyz_to_se3_poses
from pytransform3d.transform_manager import TransformManager


'''
ROS2 VILENS: runtime_drs/config/frnXXX 
Dataclass: IMU / LIDAR/ Camera
Sensor is the main class that stores sensor parameters and transformations 
'''
def get_transformation_matrix(T_AB_t_xyz_q_xyzw):
    assert len(T_AB_t_xyz_q_xyzw) == 7, f"only got {len(T_AB_t_xyz_q_xyzw)} params"
    t_xyz = T_AB_t_xyz_q_xyzw[:3]
    q_xyzw = T_AB_t_xyz_q_xyzw[3:]
    q_wxyz = [q_xyzw[-1]] + q_xyzw[:-1]
    return xyz_quat_wxyz_to_se3_poses([t_xyz], [q_wxyz])[0]

@dataclass
class IMU:
    topic: str
    rate: float
    accelerometer_noise_sigma: float
    accelerometer_bias_random_walk_sigma: float
    gyroscope_noise_sigma: float
    gyroscope_bias_random_walk_sigma: float
    W_g: List[float]

@dataclass
class LiDAR:
    B_r_BL: List[float]
    B_q_BL: List[float]
    model: str
    topic: str
    rate: float
    raw: bool
    timeshift: float

    def __post_init__(self):
        self.T_base_lidar = get_transformation_matrix(self.B_r_BL + self.B_q_BL)

@dataclass
class Camera:
    label: str
    image_height: int
    image_width: int
    rect_topic: str
    rect_intrinsics: List[float]    # 9 params
    rect_extra_params: List[float]  # 4 params
    fisheye_topic: str 
    fisheye_intrinsics: List[float]
    fisheye_extra_params: List[float]
    C_r_CL: List[float]
    C_q_CL: List[float]
    C_r_CI: List[float] = field(default_factory=list)
    C_q_CI: List[float] = field(default_factory=list)

    def __post_init__(self):
        self.T_cam_lidar = get_transformation_matrix(self.C_r_CL + self.C_q_CL)
        self.T_cam_imu = (
            get_transformation_matrix(self.C_r_CI + self.C_q_CI)
            if self.C_r_CI and self.C_q_CI else None
        )

    def get_K(self, fisheye=False):
        if fisheye:
            K_matrix = np.array(self.fisheye_intrinsics).reshape(3, 3)
        else:
            K_matrix = np.array(self.rect_intrinsics).reshape(3, 3)

        fx = K_matrix[0, 0]
        fy = K_matrix[1, 1]
        cx = K_matrix[0, 2]
        cy = K_matrix[1, 2]

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ])

@dataclass
class Sensor:
    imu: IMU
    lidar: LiDAR
    cameras: List[Camera] = field(default_factory=list)
    tf: TransformManager = field(init=False)

    def __post_init__(self):
        self.tf = TransformManager()
        self.tf.add_transform("base", "lidar", self.lidar.T_base_lidar)
        for camera in self.cameras:
            self.tf.add_transform("lidar", camera.label, camera.T_cam_lidar)
            if camera.T_cam_imu is not None and camera.label == "cam1":
                self.tf.add_transform("imu", camera.label, camera.T_cam_imu)

    @classmethod
    def load_config(cls, yaml_path: Path):
        with open(yaml_path, 'r') as file:
            data = yaml.safe_load(file)['/**']['ros__parameters']['sensors']

        imu = IMU(
            topic=data['imu']['topic'],
            rate=data['imu']['rate'],
            accelerometer_noise_sigma=data['imu']['accelerometer_noise_sigma'],
            accelerometer_bias_random_walk_sigma=data['imu']['accelerometer_bias_random_walk_sigma'],
            gyroscope_noise_sigma=data['imu']['gyroscope_noise_sigma'],
            gyroscope_bias_random_walk_sigma=data['imu']['gyroscope_bias_random_walk_sigma'],
            W_g=data['imu']['W_g'],
        )

        lidar = LiDAR(
            B_r_BL=data['lidar']['B_r_BL'],
            B_q_BL=data['lidar']['B_q_BL'],
            model=data['lidar']['model'],
            topic=data['lidar']['topic'],
            rate=data['lidar']['rate'],
            raw=data['lidar']['raw'],
            timeshift=data['lidar']['timeshift'],
        )

        cameras = []
        for cam_label, cam_data in data.items():
            if cam_label.startswith("cam"):
                cameras.append(Camera(
                    label=cam_label,
                    image_height=cam_data['image_height'],
                    image_width=cam_data['image_width'],
                    rect_topic=cam_data['rect']['topic'],
                    rect_intrinsics=cam_data['rect']['K'],
                    rect_extra_params=cam_data['rect']['D'],
                    fisheye_topic=cam_data['fisheye']['topic'],
                    fisheye_intrinsics=cam_data['fisheye']['K'],
                    fisheye_extra_params=cam_data['fisheye']['D'],
                    C_r_CL=cam_data['C_r_CL'],
                    C_q_CL=cam_data['C_q_CL'],
                    C_r_CI=cam_data.get('C_r_CI', []),
                    C_q_CI=cam_data.get('C_q_CI', []),
                ))

        return cls(imu=imu, lidar=lidar, cameras=cameras)

    def get_sensor_param(self, sensor_type: str, sensor_label: str):
        if sensor_type == "imu":
            return self.imu
        elif sensor_type == "lidar":
            return self.lidar
        elif sensor_type == "camera":
            for camera in self.cameras:
                if camera.label == sensor_label:
                    return camera
        else:
            raise ValueError(f"Unknown sensor type: {sensor_type}")


if __name__ == "__main__":
    '''
    Sensor is the internal class that stores Camera, IMU, LiDAR parameters and transformations between them
    '''
    yaml_path = Path("/home/haedam/git/runtime_drs/config/frn019/sensors_2024_10_29/sensors_frn019.yaml")
    sensor = Sensor.load_config(yaml_path)
    
    # querying extrinsics
    T_base_imu = sensor.tf.get_transform("imu", "base")
    T_base_lidar = sensor.tf.get_transform("lidar", "base")
    T_base_cam0 = sensor.tf.get_transform("cam0", "base")
    T_base_cam1 = sensor.tf.get_transform("cam1", "base")
    T_base_cam2 = sensor.tf.get_transform("cam2", "base")
    # printing transformations
    print("T_base_imu:", T_base_imu)
    print("T_base_lidar:", T_base_lidar)
    print("T_base_cam0:", T_base_cam0)
    print("T_base_cam1:", T_base_cam1)
    print("T_base_cam2:", T_base_cam2)
    
    
    # querying intrinsics
    K_fisheye_cam0 = sensor.get_sensor_param("camera", "cam0").get_K(fisheye=True)
    K_fisheye_cam1 = sensor.get_sensor_param("camera", "cam1").get_K(fisheye=True)
    K_fisheye_cam2 = sensor.get_sensor_param("camera", "cam2").get_K(fisheye=True)
    print("K_fisheye_cam0:", K_fisheye_cam0)
    print("K_fisheye_cam1:", K_fisheye_cam1)
    print("K_fisheye_cam2:", K_fisheye_cam2)
    
    K_rect_cam0 = sensor.get_sensor_param("camera", "cam0").get_K(fisheye=False)
    K_rect_cam1 = sensor.get_sensor_param("camera", "cam1").get_K(fisheye=False)
    K_rect_cam2 = sensor.get_sensor_param("camera", "cam2").get_K(fisheye=False)
    print("K_rect_cam0:", K_rect_cam0)
    print("K_rect_cam1:", K_rect_cam1)
    print("K_rect_cam2:", K_rect_cam2)