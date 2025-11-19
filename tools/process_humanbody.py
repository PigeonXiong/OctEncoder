import os
import math
import trimesh
import numpy as np
import cyminiball  # Optional; only used in the original normalization stage.
from tqdm import tqdm
from typing import Optional
from plyfile import PlyData, PlyElement
import json
import fpsample


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
        point_cloud_types += [('label', 'i4')]

    point_cloud = np.concatenate(point_cloud, axis=1)
    vertices = [tuple(p) for p in point_cloud]
    structured_array = np.array(vertices, dtype=point_cloud_types)

    el = PlyElement.describe(structured_array, 'vertex')
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    PlyData([el], text).write(filename)


def map_vertex_labels(mesh: trimesh.Trimesh, face_labels: np.ndarray) -> np.ndarray:
    """
    Given a triangular mesh and per-face labels, compute per-vertex labels
    by taking the most frequent label among faces sharing each vertex.
    """
    num_vertices = mesh.vertices.shape[0]
    vertex_labels = np.zeros(num_vertices, dtype=np.int32)

    # Build vertex → incident faces map
    vertex_faces = [[] for _ in range(num_vertices)]
    for face_idx, face in enumerate(mesh.faces):
        for vid in face:
            vertex_faces[vid].append(face_idx)

    # Assign labels by majority vote
    for vid, faces in enumerate(vertex_faces):
        if len(faces) == 0:
            vertex_labels[vid] = -1  # No adjacent face; optional fallback
        else:
            labels = face_labels[faces]
            vertex_labels[vid] = np.bincount(labels).argmax()

    return vertex_labels


def convert_mesh_to_points(input_folder: str, output_folder: str, sample_num: int = 50000, normalize: bool = True, method: str = 'surface'):
    """
    Converts ordinary triangular meshes (.obj, .ply, etc.) to sampled point clouds with segmentation labels.

    For each mesh in the input folder:
      1. Loads the mesh using trimesh.
      2. Loads corresponding segmentation JSON file (same name as mesh, .json extension).
         The JSON should map face indices to class labels.
      3. Samples points uniformly from the surface using trimesh.sample.sample_surface.
      4. Assigns a label to each point based on its source face.
      5. Normalizes the points to [-1, 1]^3 if requested.
      6. Saves the point cloud (with normals and labels) as a PLY file.

    Parameters:
        input_folder: str, path to the folder containing meshes and JSON files.
        output_folder: str, path to save generated point clouds.
        sample_num: int, number of points to sample per mesh.
        normalize: str, 'true' or 'false' for normalization.
        method: Sampling method ('surface' or 'fps').
    """
    os.makedirs(output_folder, exist_ok=True)

    mesh_files = [f for f in os.listdir(input_folder) if f.endswith('.obj') or f.endswith('.ply') or f.endswith('.stl')]
    
    for mesh_file in tqdm(mesh_files, desc="Processing meshes"):
        mesh_path = os.path.join(input_folder, mesh_file)
        json_path = os.path.join(input_folder, os.path.splitext(mesh_file)[0] + '.json')

        try:
            mesh = trimesh.load(mesh_path, process=False)
        except Exception as e:
            print(f"Skipping {mesh_file}: {e}")
            continue

        if not mesh.is_watertight:
            print(f"Warning: Mesh {mesh_file} is not watertight. Continuing...")

        # Load segmentation labels from JSON
        if os.path.exists(json_path):
            with open(json_path, 'r') as f:
                seg_data = json.load(f)
            if "sub_labels" in seg_data:
                face_labels = np.array(seg_data["sub_labels"], dtype=np.int32)  # (num_faces,)
            else:
                print(f"'sub_labels' key not found in {json_path}. Using 0 for all faces.")
                face_labels = np.zeros(len(mesh.faces), dtype=np.int32)
        else:
            print(f"No JSON for {mesh_file}. Using 0 for all faces.")
            face_labels = np.zeros(len(mesh.faces), dtype=np.int32)

        if method == 'surface':
            # Sample points from surface
            points, face_indices = trimesh.sample.sample_surface(mesh, sample_num)
            normals = mesh.face_normals[face_indices]
            labels = face_labels[face_indices].reshape(-1, 1)  # (num_points, 1)
        elif method == 'fps':
            # Sample points using farthest point sampling
            all_points = mesh.vertices
            vertex_labels = map_vertex_labels(mesh, face_labels)
            if all_points.shape[0] < sample_num:
                points = all_points
                labels = vertex_labels.reshape(-1, 1)
            else:
                kdline_fps_samples_idx = fpsample.bucket_fps_kdline_sampling(all_points, sample_num, h=7)
                kdline_fps_samples_idx = kdline_fps_samples_idx.astype(np.int32)
                points = all_points[kdline_fps_samples_idx]
                labels = vertex_labels[kdline_fps_samples_idx].reshape(-1, 1)
            normals = np.zeros_like(points)  # No normals for FPS sampling
        

        # Normalize points
        if normalize:
            bbmin = points.min(axis=0)
            bbmax = points.max(axis=0)
            center = (bbmin + bbmax) / 2
            radius = (bbmax - bbmin).max() * 0.5 + 1e-6
            points = (points - center) / radius

        # Save to PLY
        output_path = os.path.join(output_folder, os.path.splitext(mesh_file)[0] + '.ply')
        save_points_to_ply(output_path, points, normals, labels=labels)
        #print(f"Saved: {output_path}")

# -------------------------
# Run the dataset preparation for all folders
# -------------------------
def prepare_dataset(root_folder: str, sample_num: int = 4096, normalize: bool = True):
    """
    Prepares the dataset by converting all subfolders using both methods: 'surface' and 'fps'.
    """
    for subdir in os.listdir(root_folder):
        subdir_path = os.path.join(root_folder, subdir)
        if os.path.isdir(subdir_path):
            print(f"Processing folder: {subdir}")
            output_folder_surface = os.path.join(subdir_path, 'surface')
            output_folder_fps = os.path.join(subdir_path, 'fps')

            # Convert using surface sampling
            convert_mesh_to_points(
                input_folder=subdir_path,
                output_folder=output_folder_surface,
                sample_num=sample_num,
                normalize=normalize,
                method='surface'
            )

            # Convert using FPS sampling
            convert_mesh_to_points(
                input_folder=subdir_path,
                output_folder=output_folder_fps,
                sample_num=sample_num,
                normalize=normalize,
                method='fps'
            )

# Example usage
prepare_dataset("HumanBody-NS-256-3/train", sample_num=4096, normalize=True)