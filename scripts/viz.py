import open3d as o3d

# 1. Load your point cloud (adjust the file path and format as needed)

# Load the colored point cloud
pcd = o3d.io.read_point_cloud("/home/local/ASURITE/mfarazi/Mohammad/ColabNotebooks/ColabNotebooks/octformer/data/tet/points/m002S5230L062713S63TCF/m002S5230L062713S63TCF.ply")

# Check if the point cloud has colors
if not pcd.has_colors():
    print("Warning: Point cloud does not have color. Assigning random colors.")
    pcd.paint_uniform_color([0.5, 0.5, 0.5])  # Assign a neutral gray if no color is found

# Create an octree with a specified max depth
octree = o3d.geometry.Octree(max_depth=11)

# Convert the point cloud to an octree
octree.convert_from_point_cloud(pcd, size_expand=0.01)

# Visualize both the point cloud (with color) and the octree
o3d.visualization.draw_geometries([pcd, octree])
