#!/usr/bin/env python
# --------------------------------------------------------
# TetMesh Dataset Preparation for 3D Point Clouds
# (Adapted for a structure with many subfolders under data/tet/points)
# --------------------------------------------------------

import os
import math
import argparse
import trimesh
import numpy as np
import cyminiball
from tqdm import tqdm
from plyfile import PlyElement, PlyData
from typing import Optional

# -------------------------
# Function: save_points_to_ply
# -------------------------
def save_points_to_ply(filename: str, points: np.ndarray,
                       normals: Optional[np.ndarray] = None,
                       colors: Optional[np.ndarray] = None,
                       labels: Optional[np.ndarray] = None,
                       text: bool = False):
    point_cloud = [points]
    point_cloud_types = [('x', 'f4'), ('y', 'f4'), ('z', 'f4')]
    if normals is not None:
        point_cloud.append(normals)
        point_cloud_types += [('nx', 'f4'), ('ny', 'f4'), ('nz', 'f4')]
    if colors is not None:
        point_cloud.append(colors)
        point_cloud_types += [('red', 'u1'), ('green', 'u1'), ('blue', 'u1')]
    if labels is not None:
        point_cloud.append(labels)
        point_cloud_types += [('label', 'u1')]
    point_cloud = np.concatenate(point_cloud, axis=1)
    vertices = [tuple(p) for p in point_cloud]
    structured_array = np.array(vertices, dtype=point_cloud_types)
    el = PlyElement.describe(structured_array, 'vertex')
    folder = os.path.dirname(filename)
    if not os.path.exists(folder):
        os.makedirs(folder)
    PlyData([el], text).write(filename)

# -------------------------
# Argument Parsing
# -------------------------
parser = argparse.ArgumentParser(description='Prepare tetmesh dataset by sampling point clouds.')
parser.add_argument('--run', type=str, required=False, default='prepare_dataset',
                    help='The command to run.')
parser.add_argument('--sample_num', type=int, default=50000,
                    help='Number of points to sample from each mesh surface.')
parser.add_argument('--align_y', type=str, required=False, default='false',
                    help='Align the points with the y-axis ("true" or "false").')
parser.add_argument('--normalize', type=str, required=False, default='true',
                    help='Normalize the sampled points ("true" or "false").')
args = parser.parse_args()

# -------------------------
# Configuration: Set your data paths
# -------------------------
# Set the root folder where your tetmesh data is located.
# For example, update this path to point to your tetmesh directory.
# In your case, your files are under data/tet/points.
root_folder = "data/tet"
# Use root_folder as the input folder.
input_folder = root_folder  
# Define the folder to save the generated PLY files.
output_folder = input_folder + ".ply"

# -------------------------
# Helper Functions for TetMesh Processing
# -------------------------
def read_tetgen_files(folder_path: str):
    """
    Reads TetGen files from the given folder.
    Expects one .node file and one .ele file.
    Returns:
      vertices: np.ndarray of shape (N, 3)
      tets: np.ndarray of shape (M, 4) (tetrahedra connectivity, 0-indexed)
    """
    node_file = None
    ele_file = None
    for f in os.listdir(folder_path):
        if f.endswith('.node'):
            node_file = os.path.join(folder_path, f)
        elif f.endswith('.ele'):
            ele_file = os.path.join(folder_path, f)
    if node_file is None or ele_file is None:
        raise ValueError(f"Missing .node or .ele file in folder {folder_path}")
    # Read .node file
    with open(node_file, 'r') as fid:
        lines = fid.readlines()
    header = lines[0].strip().split()
    num_nodes = int(header[0])
    vertices = []
    for line in lines[1:num_nodes+1]:
        parts = line.strip().split()
        vertices.append([float(parts[1]), float(parts[2]), float(parts[3])])
    vertices = np.array(vertices)
    # Read .ele file
    with open(ele_file, 'r') as fid:
        lines = fid.readlines()
    header = lines[0].strip().split()
    num_tets = int(header[0])
    tets = []
    for line in lines[1:num_tets+1]:
        parts = line.strip().split()
        tet = [int(p) for p in parts[1:5]]
        tet = [i - 1 for i in tet]  # Convert to 0-indexed.
        tets.append(tet)
    tets = np.array(tets)
    return vertices, tets

def extract_boundary_faces(tets):
    """
    Extracts boundary (surface) faces from tetrahedral connectivity.
    Each tetrahedron has 4 faces; faces that appear only once are boundary faces.
    Returns:
      boundary_faces: np.ndarray of shape (K, 3)
    """
    face_dict = {}
    for tet in tets:
        faces = [
            tuple(sorted([tet[0], tet[1], tet[2]])),
            tuple(sorted([tet[0], tet[1], tet[3]])),
            tuple(sorted([tet[0], tet[2], tet[3]])),
            tuple(sorted([tet[1], tet[2], tet[3]]))
        ]
        for face in faces:
            face_dict[face] = face_dict.get(face, 0) + 1
    boundary_faces = [face for face, count in face_dict.items() if count == 1]
    return np.array(boundary_faces)

def _get_point_folder():
    """
    Constructs the folder name for the sampled point clouds.
    Here, we simply use output_folder as defined above.
    """
    return output_folder

def convert_tetmesh_to_points():
    """
    For each tetmesh sample in input_folder, extracts boundary faces,
    samples points on the surface, optionally normalizes them,
    and saves the result as a PLY file.
    """
    os.makedirs(output_folder, exist_ok=True)
    subfolders = [d for d in os.listdir(input_folder) if os.path.isdir(os.path.join(input_folder, d))]
    sample_num = args.sample_num
    ply_folder = _get_point_folder()
    print("-> Sampling points on meshes.")
    for sub in tqdm(subfolders, desc="Processing tetmesh samples", ncols=80):
        folder_path = os.path.join(input_folder, sub)
        try:
            vertices, tets = read_tetgen_files(folder_path)
        except Exception as e:
            print(f"Skipping {folder_path}: {e}")
            continue
        boundary_faces = extract_boundary_faces(tets)
        if len(boundary_faces) == 0:
            print(f"No boundary faces found in {folder_path}")
            continue
        mesh = trimesh.Trimesh(vertices=vertices, faces=boundary_faces, process=False)
        points, face_indices = trimesh.sample.sample_surface(mesh, sample_num)
        normals = mesh.face_normals[face_indices]
        if args.normalize.lower() == 'true':
            bbmin = points.min(axis=0)
            bbmax = points.max(axis=0)
            center = (bbmin + bbmax) / 2
            radius = (bbmax - bbmin).max() * 0.5 + 1e-6
            points = (points - center) / radius
        filename_ply = os.path.join(ply_folder, sub + '.ply')
        save_points_to_ply(filename_ply, points, normals)

# -------------------------
# File List Generation for All Samples
# -------------------------
def get_filelist_all(root_folder, suffix='ply', ratio=1.0):
    """
    Traverses the root_folder (which is the folder where PLY files are saved)
    and collects relative paths of PLY files from all subfolders.
    Returns:
      filelist: List of relative file paths.
      labels: List of default labels (set to -1, since no binary labels exist).
    """
    filelist = []
    labels = []
    folders = sorted(os.listdir(root_folder))
    for folder in folders:
        full_folder = os.path.join(root_folder, folder)
        if not os.path.isdir(full_folder):
            continue
        filenames = sorted([f for f in os.listdir(full_folder) if f.endswith(suffix)])
        total_num = math.ceil(len(filenames) * ratio)
        for i in range(total_num):
            rel_path = os.path.join(folder, filenames[i])
            filelist.append(rel_path)
            labels.append(-1)  # Default label; adjust if needed.
    return filelist, labels

def generate_points_filelist():
    """
    Generates a file list for all PLY files under the points folder.
    The list is saved in a folder named "filelist" located in the parent directory of input_folder.
    """
    print("-> Generating file list")
    points_folder = _get_point_folder()
    list_folder = os.path.join(os.path.dirname(input_folder), 'filelist')
    if not os.path.exists(list_folder):
        os.makedirs(list_folder)
    # Generate full file list.
    filelist, labels = get_filelist_all(points_folder, suffix='ply', ratio=1.0)
    filename = os.path.join(list_folder, 'all_samples.txt')
    print("Saving file list to:", filename)
    with open(filename, 'w') as fid:
        for f, lab in zip(filelist, labels):
            fid.write(f"{f} {lab}\n")
    # Optionally, generate additional file lists for various ratios.
    ratios = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0]
    for ratio in ratios:
        filelist, labels = get_filelist_all(points_folder, suffix='ply', ratio=ratio)
        filename = os.path.join(list_folder, f'all_{ratio:.02f}.txt')
        print("Saving file list (ratio", ratio, ") to:", filename)
        with open(filename, 'w') as fid:
            for f, lab in zip(filelist, labels):
                fid.write(f"{f} {lab}\n")

# -------------------------
# Main function: Prepare Dataset
# -------------------------
def prepare_dataset():
    convert_tetmesh_to_points()
    generate_points_filelist()

# -------------------------
# Main Execution
# -------------------------
if __name__ == '__main__':
    eval(f'{args.run}()')
