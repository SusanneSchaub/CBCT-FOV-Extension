import torch


# def get_voxel_indices(points, voxels, batch_size=10000):
#     # points: [n_points, 3]
#     voxel_shape = voxels.shape[:3]

#     # Assuming voxels are uniformly spaced, calculate voxel spacing along each dimension
#     voxel_spacing_x = voxels[1, 0, 0, 0] - voxels[0, 0, 0, 0]
#     voxel_spacing_y = voxels[0, 1, 0, 1] - voxels[0, 0, 0, 1]
#     voxel_spacing_z = voxels[0, 0, 1, 2] - voxels[0, 0, 0, 2]

#     voxel_indices_list = []

#     # Process points in batches to manage memory usage
#     for start in range(0, points.shape[0], batch_size):
#         end = min(start + batch_size, points.shape[0])
#         x, y, z = points[start:end, 0], points[start:end, 1], points[start:end, 2]

#         # Calculate voxel indices for each point (vectorized)
#         i = torch.floor((x - voxels[0, 0, 0, 0]) / voxel_spacing_x).long()
#         j = torch.floor((y - voxels[0, 0, 0, 1]) / voxel_spacing_y).long()
#         k = torch.floor((z - voxels[0, 0, 0, 2]) / voxel_spacing_z).long()

#         # Clip indices to ensure they are within the voxel grid
#         i = torch.clamp(i, 0, voxel_shape[0] - 2)
#         j = torch.clamp(j, 0, voxel_shape[1] - 2)
#         k = torch.clamp(k, 0, voxel_shape[2] - 2)

#         # Generate the 8 vertices for each point (shape: [batch_size, 8, 3])
#         voxel_indices_batch = torch.stack([
#             torch.stack([i, j, k], dim=-1),
#             torch.stack([i + 1, j, k], dim=-1),
#             torch.stack([i, j + 1, k], dim=-1),
#             torch.stack([i, j, k + 1], dim=-1),
#             torch.stack([i + 1, j + 1, k], dim=-1),
#             torch.stack([i + 1, j, k + 1], dim=-1),
#             torch.stack([i, j + 1, k + 1], dim=-1),
#             torch.stack([i + 1, j + 1, k + 1], dim=-1)
#         ], dim=1)

#         # Move the batch to CPU to free up GPU memory and append to the list
#         voxel_indices_list.append(voxel_indices_batch.cpu())

#     # Concatenate all batches on the CPU
#     voxel_indices = torch.cat(voxel_indices_list, dim=0)
#     return voxel_indices




def get_voxel_indices(points, voxels):
    # points: [n_points, 3]
    x, y, z = points[:, 0], points[:, 1], points[:, 2]
    voxel_shape = voxels.shape[:3]

    # Assuming voxels are uniformly spaced, calculate voxel spacing along each dimension
    voxel_spacing_x = voxels[1, 0, 0, 0] - voxels[0, 0, 0, 0]
    voxel_spacing_y = voxels[0, 1, 0, 1] - voxels[0, 0, 0, 1]
    voxel_spacing_z = voxels[0, 0, 1, 2] - voxels[0, 0, 0, 2]

    # Calculate voxel indices for each point (vectorized)
    i = torch.floor((x - voxels[0, 0, 0, 0]) / voxel_spacing_x).long()
    j = torch.floor((y - voxels[0, 0, 0, 1]) / voxel_spacing_y).long()
    k = torch.floor((z - voxels[0, 0, 0, 2]) / voxel_spacing_z).long()

    # Clip indices to ensure they are within the voxel grid
    i = torch.clamp(i, 0, voxel_shape[0] - 2)
    j = torch.clamp(j, 0, voxel_shape[1] - 2)
    k = torch.clamp(k, 0, voxel_shape[2] - 2)

    # Generate the 8 vertices for each point (shape: [n_points, 8, 3])
    voxel_indices = torch.stack([
        torch.stack([i, j, k], dim=-1),
        torch.stack([i + 1, j, k], dim=-1),
        torch.stack([i, j + 1, k], dim=-1),
        torch.stack([i, j, k + 1], dim=-1),
        torch.stack([i + 1, j + 1, k], dim=-1),
        torch.stack([i + 1, j, k + 1], dim=-1),
        torch.stack([i, j + 1, k + 1], dim=-1),
        torch.stack([i + 1, j + 1, k + 1], dim=-1)
    ], dim=1)

    return voxel_indices



def get_nearest_neighbor_value(points, vertices, values):
    # points shape: [n_points, 3]
    # vertices shape: [n_points, 8, 3]
    # values shape: [n_points, 8]

    # Compute the squared Euclidean distances between each point and its 8 vertices
    # Distance formula: ||p - v||^2 = (p_x - v_x)^2 + (p_y - v_y)^2 + (p_z - v_z)^2
    distances = torch.sum((points[:, None, :] - vertices) ** 2, dim=-1)  # shape: [196608, 8]

    nearest_vertex_indices = torch.argmin(distances, dim=1)  # shape: [196608]
    nearest_neighbor_values = values[torch.arange(values.shape[0]), nearest_vertex_indices]  # shape: [196608]

    return nearest_neighbor_values



def get_trilinear_interpolation_value(points, vertices, values):
    # points shape: [n_points, 3]
    # vertices shape: [n_points, 8, 3]
    # values shape: [n_points, 8]
    
    # Separate the vertices into the corners (assume that vertices[0] is the (0,0,0) corner and vertices[7] is the (1,1,1) corner)
    x0, y0, z0 = vertices[:, 0, 0], vertices[:, 0, 1], vertices[:, 0, 2]  # Lower-left (0, 0, 0) vertex of each cubic
    x1, y1, z1 = vertices[:, 7, 0], vertices[:, 7, 1], vertices[:, 7, 2]  # Upper-right (1, 1, 1) vertex of each cubic
    
    # Get the values for the 8 vertices
    v000, v100, v010, v110, v001, v101, v011, v111 = values[:, 0], values[:, 1], values[:, 2], values[:, 3], values[:, 4], values[:, 5], values[:, 6], values[:, 7]
    
    # Extract the point coordinates
    px, py, pz = points[:, 0], points[:, 1], points[:, 2]
    
    # Calculate interpolation weights
    tx = (px - x0) / (x1 - x0)
    ty = (py - y0) / (y1 - y0)
    tz = (pz - z0) / (z1 - z0)
    
    # Perform trilinear interpolation, Output shape: [n_points]
    interpolated_values = (
        (1 - tx) * (1 - ty) * (1 - tz) * v000 +
        tx * (1 - ty) * (1 - tz) * v100 +
        (1 - tx) * ty * (1 - tz) * v010 +
        tx * ty * (1 - tz) * v110 +
        (1 - tx) * (1 - ty) * tz * v001 +
        tx * (1 - ty) * tz * v101 +
        (1 - tx) * ty * tz * v011 +
        tx * ty * tz * v111
    )
    
    return interpolated_values  



def matcher(points, voxels, imgFDK, mode='mean'):
    # points shape: [n_rays, n_samples, 3] / [128,128,128,3]
    # voxels shape: [128, 128, 128, 3]
    # imgFDK shape: [128, 128, 128]

    shape = points.shape
    points = points.reshape(-1, 3)      # n_points = n_rays*n_samples

    indices = get_voxel_indices(points, voxels)     # [n_points, 8, 3]

    x, y, z = indices[..., 0], indices[..., 1], indices[..., 2]
    values = imgFDK[x, y, z]        # [n_points, 8]
    vertices = voxels[x, y, z]      # [n_points, 8, 3]

    if mode == 'nearest':
        rhos = get_nearest_neighbor_value(points, vertices, values)
    elif mode == 'trilinear':
        rhos = get_trilinear_interpolation_value(points, vertices, values)
    else:
        rhos = torch.mean(values, dim=-1)
    
    rhos = rhos.reshape(shape[:-1])      # [b, n_rays, n_samples]
    rhos = rhos.unsqueeze(-1)           # [b, n_rays, n_samples, 1] 
    return rhos


