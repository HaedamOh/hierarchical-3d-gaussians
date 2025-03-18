import os
import numpy as np


# VERTEX_SE3:QUAT_TIME id x y z qx qy qz qw sec nsec
# T_map_base
def readNodes_base(file_path):
    frames = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("VERTEX_SE3"):
                parts = line.split(" ")
                vertex_id = parts[1]
                sec = parts[9]  # Keep sec as string don't remove zero
                # TODO issue: zero infront of nsec removed
                nsec = (
                    parts[10].strip().zfill(9)
                )  # remove any trailing whitespace/newline

                # Create a dictionary for each pose
                param = {
                    "id": vertex_id,
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "z": float(parts[4]),
                    "qx": float(parts[5]),
                    "qy": float(parts[6]),
                    "qz": float(parts[7]),
                    "qw": float(parts[8]),
                    "sec": sec,
                    "nsec": nsec,
                    "timestamp": f"{sec}.{nsec}",
                    "cloud_name": f"cloud_{sec}_{nsec}.pcd",
                }
                frames[vertex_id] = param
    return frames


# IMAGE_POSE: image_id x y z qx qy qz qw image_sec image_nsec
# is also T_map_base of sensor at time i when that image is overlayed
def readNodes_image(file_path):
    frames = {}
    with open(file_path, "r") as f:
        for line in f:
            if line.startswith("IMAGE_POSE"):
                parts = line.split(" ")
                image_id = parts[1]
                sec = parts[9]  # Keep sec as string don't remove zero
                # TODO issue: zero infront of nsec removed
                nsec = (
                    parts[10].strip().zfill(9)
                )  # Keep nsec as string and remove any trailing whitespace/newline

                # Create a dictionary for each pose
                pose = {
                    "id": image_id,
                    "x": float(parts[2]),
                    "y": float(parts[3]),
                    "z": float(parts[4]),
                    "qx": float(parts[5]),
                    "qy": float(parts[6]),
                    "qz": float(parts[7]),
                    "qw": float(parts[8]),
                    "sec": sec,
                    "nsec": nsec,
                    "timestamp": f"{sec}.{nsec}",
                    "image_name": f"image_{sec}_{nsec}.jpg",
                }
                frames[image_id] = pose
    return frames
