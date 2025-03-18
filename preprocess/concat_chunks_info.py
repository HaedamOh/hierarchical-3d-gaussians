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

import argparse
import os
import numpy as np

# if __name__ == 'main':
parser = argparse.ArgumentParser()
parser.add_argument('--base_dir', default='./example_dataset/camera_calibration/chunks', help="Chunks folder")
parser.add_argument('--dest_dir', default='./example_dataset/camera_calibration/aligned', help="Folder to which chunks.txt file will be written")
args = parser.parse_args()

chunks = os.listdir(args.base_dir)

chunks_data = []
for chunk in chunks:
    center_file_path = os.path.join(args.base_dir, chunk + "/center.txt")
    extents_file_path = os.path.join(args.base_dir, chunk + "/extent.txt")

    chunk = {
        "name":chunk,
        "center": [0,0,0],
        "extent": [0,0,0]
    }

    try:
        with open(center_file_path, 'r') as file:
            content = file.read()
            chunk["center"] = content.split(" ")
    except FileNotFoundError:
        print(f"File not found: {center_file_path}")

    try:
        with open(extents_file_path, 'r') as file:
            content = file.read()
            chunk["extent"] = content.split(" ")
    except FileNotFoundError:
        print(f"File not found: {extents_file_path}")

    chunks_data.append(chunk)

def write_chunks(data, output_directory):
    file_path = os.path.join(output_directory, "chunks.txt")
    try:
        with open(file_path, 'w') as file:
            ind = 0
            for chunk in data:
                line = chunk['name'] + " " + ' '.join(map(str, chunk['center'])) + " " +' '.join(map(str, chunk['extent'])) + "\n"
                
                if ind == len(data)-1:
                    line = line[:-1]

                # Write content to the file
                file.write(line)
                ind += 1
            print(f"Content written to {file_path}")

    except IOError:
        print(f"Error writing to {file_path}")

write_chunks(chunks_data, args.dest_dir)

# Also save center.txt and extent.txt files in dest_dir
centers = np.array([chunk['center'] for chunk in chunks_data], dtype=np.float32)  # (n, 3)
extents = np.array([chunk['extent'] for chunk in chunks_data], dtype=np.float32)  # (n, 3)

min_bounds = np.min(centers - extents/2, axis=0)
max_bounds = np.max(centers + extents/2, axis=0)

overall_center = (min_bounds + max_bounds) / 2 # (3,)
overall_extent = max_bounds - min_bounds # (3,)
overall_center_file_path = os.path.join(args.dest_dir, "center.txt")
overall_extent_file_path = os.path.join(args.dest_dir, "extent.txt")

try:
    with open(overall_center_file_path, 'w') as file:
        file.write(' '.join(map(str, overall_center)))
    print(f"Content written to {overall_center_file_path}")
except IOError:
    print(f"Error writing to {overall_center_file_path}")

try:
    with open(overall_extent_file_path, 'w') as file:
        file.write(' '.join(map(str, overall_extent)))
    print(f"Content written to {overall_extent_file_path}")
except IOError:
    print(f"Error writing to {overall_extent_file_path}")

