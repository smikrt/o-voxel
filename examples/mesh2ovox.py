import torch
import o_voxel
import utils
import trimesh
import trimesh.visual
import polyscope as ps 
import numpy as np
import polyscope.imgui as psim

ps.init()

RES = 128

asset = trimesh.load("/mnt/c/Users/tkimt/coding/TRELLIS.2/o-voxel/examples/cache/pole.glb")

# 0. Normalize asset to unit cube
aabb = asset.bounding_box.bounds
center = (aabb[0] + aabb[1]) / 2
scale = 0.99999 / (aabb[1] - aabb[0]).max()     # To avoid numerical issues
asset.apply_translation(-center)
asset.apply_scale(scale)

# 1. Geometry Voxelization (Flexible Dual Grid)
# Returns: occupied indices, dual vertices (QEF solution), and edge intersected
mesh = asset.to_mesh()
vertices = torch.from_numpy(mesh.vertices).float()
faces = torch.from_numpy(mesh.faces).long()
voxel_indices, dual_vertices, intersected = o_voxel.convert.mesh_to_flexible_dual_grid(
    vertices, faces,
    grid_size=RES,                              # Resolution
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],      # Axis-aligned bounding box
    face_weight=1.0,                            # Face term weight in QEF
    boundary_weight=0.2,                        # Boundary term weight in QEF
    regularization_weight=1e-2,                 # Regularization term weight in QEF
    timing=True
)
## sort to ensure align between geometry and material voxelization
vid = o_voxel.serialize.encode_seq(voxel_indices)
mapping = torch.argsort(vid)
voxel_indices = voxel_indices[mapping]
dual_vertices = dual_vertices[mapping]
intersected = intersected[mapping]

# 2. Material Voxelization (Volumetric Attributes)
# Returns: dict containing 'base_color', 'metallic', 'roughness', etc.
voxel_indices_mat, attributes = o_voxel.convert.textured_mesh_to_volumetric_attr(
    asset,
    grid_size=RES,
    aabb=[[-0.5,-0.5,-0.5],[0.5,0.5,0.5]],
    timing=True
)
## sort to ensure align between geometry and material voxelization
vid_mat = o_voxel.serialize.encode_seq(voxel_indices_mat)
mapping_mat = torch.argsort(vid_mat)
attributes = {k: v[mapping_mat] for k, v in attributes.items()}

# Save to compressed .vxz format
## packing
dual_vertices = dual_vertices * RES - voxel_indices
dual_vertices = (torch.clamp(dual_vertices, 0, 1) * 255).type(torch.uint8)
intersected = (intersected[:, 0:1] + 2 * intersected[:, 1:2] + 4 * intersected[:, 2:3]).type(torch.uint8)
attributes['dual_vertices'] = dual_vertices
attributes['intersected'] = intersected
o_voxel.io.write("weird_artifact.vxz", voxel_indices, attributes)


# 3. Polyscope 
# Convert PyTorch tensors to NumPy for Polyscope.
voxel_np = voxel_indices.detach().cpu().numpy().astype(np.int32)
dual_np = dual_vertices.detach().cpu().numpy()
vertices_np = vertices.detach().cpu().numpy()
faces_np = faces.detach().cpu().numpy()
intersected_np = intersected.detach().cpu().numpy()
 
# original mesh
ps.register_surface_mesh(
    "input duck",
    vertices_np,
    faces_np,
    enabled=True,
    color=(0.9, 0.7, 0.2),
    smooth_shade=True
)
 
# define the grid parameters to match the voxelization
origin = (-0.5, -0.5, -0.5)
cell_width = (1.0 / RES, 1.0 / RES, 1.0 / RES)

# occupied cells from the voxelization, by their integer grid indices
occupied_cells = voxel_np

# register the grid
ps_grid = ps.register_sparse_volume_grid("occupied voxels", origin, cell_width, occupied_cells)
ps_grid.set_render_mode("wireframe")

# decode the packed intersected bits back into x/y/z flags
flags = np.stack([(intersected_np[:, 0] >> a) & 1 for a in range(3)], axis=1).astype(bool)

starts_all, ends_all = [], []
for a in range(3):
    v = voxel_np[flags[:, a]]
    offset = np.ones(3)
    offset[a] = 0          # flagged edge sits at the voxel's +1 corner in the other two axes
    step = np.zeros(3)
    step[a] = 1
    starts = np.array(origin) + (v + offset) * np.array(cell_width)
    starts_all.append(starts)
    ends_all.append(starts + step * np.array(cell_width))

starts = np.concatenate(starts_all)
ends = np.concatenate(ends_all)
n = len(starts)

nodes = np.concatenate([starts, ends], axis=0)
edges = np.stack([np.arange(n), np.arange(n) + n], axis=1)

ps.register_curve_network("crossed edges", nodes, edges, color=(1.0, 0.0, 0.0), radius=0.002)

#hermite data
points, normals = [], []
for a in range(3):
    v = voxel_np[flags[:, a]]
    if len(v) == 0:
        continue
    offset = np.ones(3)
    offset[a] = 0                      # crossed edge sits at the voxel's +1 corner in the other two axes
    starts = np.array(origin) + (v + offset) * np.array(cell_width)
    dirs = np.zeros_like(starts)
    dirs[:, a] = 1.0

    # all hits along each edge, keep only those within one cell length
    locs, ray_idx, tri_idx = mesh.ray.intersects_location(starts, dirs, multiple_hits=True)
    keep = np.linalg.norm(locs - starts[ray_idx], axis=1) <= cell_width[a]
    points.append(locs[keep])
    normals.append(mesh.face_normals[tri_idx[keep]])

hermite_p = np.concatenate(points)
hermite_n = np.concatenate(normals)

ps_h = ps.register_point_cloud("hermite points", hermite_p, radius=0.003, color=(0.0, 1.0, 0.0))
ps_h.add_vector_quantity("hermite normals", hermite_n, enabled=True, length=0.03)

# per-voxel normals from the material voxelization
mat_coords = voxel_indices_mat[mapping_mat].cpu().numpy()
centers = np.array(origin) + (mat_coords + 0.5) * np.array(cell_width)
ps_n = ps.register_point_cloud("voxel centers", centers, radius=0.002)

# dual vertices 
dual_vert = np.array(origin) + (voxel_np + dual_np / 255.0) * np.array(cell_width)
ps.register_point_cloud("dual vertices", dual_vert, radius=0.004, color=(1.0, 0.0, 1.0))

# line from each voxel center to its dual vertex, to show how far it moved
voxel_centers = np.array(origin) + (voxel_np + 0.5) * np.array(cell_width)
n_dual = len(dual_vert)
ps.register_curve_network(
    "dual offset",
    np.concatenate([voxel_centers, dual_vert]),
    np.stack([np.arange(n_dual), np.arange(n_dual) + n_dual], axis=1),
    color=(1.0, 0.0, 1.0),
    radius=0.001
)

# 4. Reconstruct mesh from the saved .vxz (same as ovox2mesh.py, with its own names)
rec_coords, rec_data = o_voxel.io.read("weird_artifact.vxz")
rec_dual = rec_data['dual_vertices'] / 255
rec_flags = torch.cat([
    rec_data['intersected'] % 2,
    rec_data['intersected'] // 2 % 2,
    rec_data['intersected'] // 4 % 2,
], dim=-1).bool()

rec_verts, rec_faces = o_voxel.convert.flexible_dual_grid_to_mesh(
    rec_coords.cuda(),
    rec_dual.cuda(),
    rec_flags.cuda(),
    split_weight=None,   # auto-split based on min angle
    grid_size=RES,
    aabb=[[-0.5, -0.5, -0.5], [0.5, 0.5, 0.5]],
)
rec_verts_np = rec_verts.cpu().numpy()
rec_faces_np = rec_faces.cpu().numpy()
rec_color = rec_data['base_color'].cpu().numpy()

# save as ply
rec_mesh = trimesh.Trimesh(
    vertices=rec_verts_np, faces=rec_faces_np,
    visual=trimesh.visual.ColorVisuals(vertex_colors=rec_color),
    process=False
)
rec_mesh.export("rec_weird_artifact.ply")

# show it
ps_rec = ps.register_surface_mesh("reconstructed duck", rec_verts_np, rec_faces_np, edge_width=1.0)

# 5. Dual quad mesh: connect the dual vertices (magenta points) into quads, one per crossed edge.
# Each crossed edge is shared by 4 voxels; their 4 dual vertices make the quad.
# Same as flexible_dual_grid_to_mesh before it splits quads into triangles.
quad_offsets = np.array([
    [[0, 0, 0], [0, 0, 1], [0, 1, 1], [0, 1, 0]],     # x-axis edge
    [[0, 0, 0], [1, 0, 0], [1, 0, 1], [0, 0, 1]],     # y-axis edge
    [[0, 0, 0], [0, 1, 0], [1, 1, 0], [1, 0, 0]],     # z-axis edge
])

# voxel coord -> row in voxel_np (and dual_vert)
key_base = RES + 2
def _vkey(c):
    c = c.astype(np.int64) + 1
    return (c[..., 0] * key_base + c[..., 1]) * key_base + c[..., 2]
voxel_keys = _vkey(voxel_np)
key_order = np.argsort(voxel_keys)
sorted_keys = voxel_keys[key_order]

quads_all = []
for a in range(3):
    v = voxel_np[flags[:, a]]
    if len(v) == 0:
        continue
    nb_keys = _vkey(v[:, None, :] + quad_offsets[a][None])           # (E, 4)
    pos = np.clip(np.searchsorted(sorted_keys, nb_keys), 0, len(sorted_keys) - 1)
    found = (sorted_keys[pos] == nb_keys).all(axis=1)                 # all 4 voxels must exist
    quads = key_order[pos]                                            # (E, 4) indices into dual_vert

    # orient each quad to match the input surface normal at the edge crossing
    # (the offset order above gives a quad facing -axis by default)
    offset = np.ones(3)
    offset[a] = 0
    dirs = np.zeros((len(v), 3))
    dirs[:, a] = 1.0
    starts = np.array(origin) + (v + offset) * np.array(cell_width)
    hit_tri, hit_ray, hit_loc = mesh.ray.intersects_id(starts, dirs, multiple_hits=False, return_locations=True)
    flip = np.zeros(len(v), dtype=bool)
    flip[hit_ray] = mesh.face_normals[hit_tri][:, a] > 0
    quads[flip] = quads[flip][:, ::-1]

    quads_all.append(quads[found])

quad_faces_np = np.concatenate(quads_all)
print(f"dual quads: {len(quad_faces_np)} quads")

ps.register_surface_mesh("dual quads", dual_vert, quad_faces_np,
                         edge_width=1.0, color=(0.85, 0.85, 0.9))

ps.show()
