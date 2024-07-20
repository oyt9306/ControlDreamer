import numpy as np
import torch
import random
import math 
import torch.nn.functional as F

def create_camera_to_world_matrix(elevation, azimuth):
    elevation = np.radians(elevation)
    azimuth = np.radians(azimuth)
    # Convert elevation and azimuth angles to Cartesian coordinates on a unit sphere
    x = np.cos(elevation) * np.sin(azimuth)
    y = np.sin(elevation)
    z = np.cos(elevation) * np.cos(azimuth)
    
    # Calculate camera position, target, and up vectors
    camera_pos = np.array([x, y, z])
    target = np.array([0, 0, 0])
    up = np.array([0, 1, 0])
    
    # Construct view matrix
    forward = target - camera_pos
    forward /= np.linalg.norm(forward)
    right = np.cross(forward, up)
    right /= np.linalg.norm(right)
    new_up = np.cross(right, forward)
    new_up /= np.linalg.norm(new_up)
    cam2world = np.eye(4)
    cam2world[:3, :3] = np.array([right, new_up, -forward]).T
    cam2world[:3, 3] = camera_pos
    return cam2world


def convert_opengl_to_blender(camera_matrix):
    if isinstance(camera_matrix, np.ndarray):
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = np.array([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        camera_matrix_blender = np.dot(flip_yz, camera_matrix)
    else:
        # Construct transformation matrix to convert from OpenGL space to Blender space
        flip_yz = torch.tensor([[1, 0, 0, 0], [0, 0, -1, 0], [0, 1, 0, 0], [0, 0, 0, 1]])
        if camera_matrix.ndim == 3:
            flip_yz = flip_yz.unsqueeze(0)
        camera_matrix_blender = torch.matmul(flip_yz.to(camera_matrix), camera_matrix)
    return camera_matrix_blender


def normalize_camera(camera_matrix):
    ''' normalize the camera location onto a unit-sphere'''
    if isinstance(camera_matrix, np.ndarray):
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (np.linalg.norm(translation, axis=1, keepdims=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    else:
        camera_matrix = camera_matrix.reshape(-1,4,4)
        translation = camera_matrix[:,:3,3]
        translation = translation / (torch.norm(translation, dim=1, keepdim=True) + 1e-8)
        camera_matrix[:,:3,3] = translation
    return camera_matrix.reshape(-1,16)

def return_random_camera(batch_size):
    ### Default setting of MVdream ###
    fovy_range =  [15, 60]
    camera_distance_range = [0.8, 1.0] 
    azimuth_range =  [-180, 180]
    fovy_range = [15, 60]
    elevation_range =  [0, 30]
    n_view = 4
    relative_radius = True
    zoom_range = (1.0, 1.0)
    real_batch_size = batch_size // n_view

    # sample elevation angles
    if random.random() < 0.5:
        # sample elevation angles uniformly with a probability 0.5 (biased towards poles)
        elevation_deg = (
            torch.rand(real_batch_size)
            * (elevation_range[1] - elevation_range[0])
            + elevation_range[0]
        ).repeat_interleave(n_view, dim=0)
        elevation = elevation_deg * math.pi / 180
    else:
        # otherwise sample uniformly on sphere
        elevation_range_percent = [
            (elevation_range[0] + 90.0) / 180.0,
            (elevation_range[1] + 90.0) / 180.0,
        ]
        # inverse transform sampling
        elevation = torch.asin(
            2
            * (
                torch.rand(real_batch_size)
                * (elevation_range_percent[1] - elevation_range_percent[0])
                + elevation_range_percent[0]
            )
            - 1.0
        ).repeat_interleave(n_view, dim=0)
        elevation_deg = elevation / math.pi * 180.0

    azimuth_deg = (
        torch.rand(real_batch_size).reshape(-1,1) + torch.arange(n_view).reshape(1,-1)
    ).reshape(-1) / n_view * (
        azimuth_range[1] - azimuth_range[0]
    ) + azimuth_range[
        0
    ]
    azimuth = azimuth_deg * math.pi / 180

    ######## Different from original ########
    # sample fovs from a uniform distribution bounded by fov_range
    fovy_deg: Float[Tensor, "B"] = (
        torch.rand(real_batch_size)
        * (fovy_range[1] - fovy_range[0])
        + fovy_range[0]
    ).repeat_interleave(n_view, dim=0)
    fovy = fovy_deg * math.pi / 180

    # sample distances from a uniform distribution bounded by distance_range
    camera_distances= (
        torch.rand(real_batch_size)
        * (camera_distance_range[1] - camera_distance_range[0])
        + camera_distance_range[0]
    ).repeat_interleave(n_view, dim=0)
    if relative_radius:
        scale = 1 / torch.tan(0.5 * fovy)
        camera_distances = scale * camera_distances

    # zoom in by decreasing fov after camera distance is fixed
    zoom: Float[Tensor, "B"] = (
        torch.rand(real_batch_size)
        * (zoom_range[1] - zoom_range[0])
        + zoom_range[0]
    ).repeat_interleave(n_view, dim=0)
    fovy = fovy * zoom
    fovy_deg = fovy_deg * zoom

    camera_positions = torch.stack(
        [
            camera_distances * torch.cos(elevation) * torch.cos(azimuth),
            camera_distances * torch.cos(elevation) * torch.sin(azimuth),
            camera_distances * torch.sin(elevation),
        ],
        dim=-1,
    )

    # default scene center at origin
    center = torch.zeros_like(camera_positions)
    # default camera up direction as +z
    up = torch.as_tensor([0, 0, 1], dtype=torch.float32)[
        None, :
    ].repeat(batch_size, 1)

    lookat = F.normalize(center - camera_positions, dim=-1)
    right  = F.normalize(torch.cross(lookat, up), dim=-1)
    up = F.normalize(torch.cross(right, lookat), dim=-1)
    c2w3x4 = torch.cat(
        [torch.stack([right, up, -lookat], dim=-1), camera_positions[:, :, None]],
        dim=-1,
    )
    c2w = torch.cat(
        [c2w3x4, torch.zeros_like(c2w3x4[:, :1])], dim=1
    )
    c2w[:, 3, 3] = 1.0
    return c2w.flatten(start_dim=1)


def get_camera(num_frames, elevation=15, azimuth_start=0, azimuth_span=360, blender_coord=True):
    angle_gap = azimuth_span / num_frames
    cameras = []
    for azimuth in np.arange(azimuth_start, azimuth_span+azimuth_start, angle_gap):
        camera_matrix = create_camera_to_world_matrix(elevation, azimuth)
        if blender_coord:
            camera_matrix = convert_opengl_to_blender(camera_matrix)
        cameras.append(camera_matrix.flatten())
    return torch.tensor(np.stack(cameras, 0)).float()