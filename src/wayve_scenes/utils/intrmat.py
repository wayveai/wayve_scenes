from __future__ import annotations

import numpy as np
import torch
from einops import repeat
from torch.utils.data import default_convert


def get_focal_lengths(intrinsics: torch.Tensor | np.ndarray):
    intrinsics = _normalize_shape_and_type(intrinsics)
    return intrinsics[:, [0, 1], [0, 1]]


def get_principal_point(intrinsics: torch.Tensor | np.ndarray):
    intrinsics = _normalize_shape_and_type(intrinsics)
    return intrinsics[:, [0, 1], 2]


def as_matrix(focal_lengths: torch.Tensor, principal_points: torch.Tensor) -> torch.Tensor:
    focal_lengths = default_convert(focal_lengths)
    principal_points = default_convert(principal_points)
    if focal_lengths.ndim == 1:
        focal_lengths = focal_lengths.unsqueeze(0)
    if principal_points.ndim == 1:
        principal_points = principal_points.unsqueeze(0)

    intrinsics = repeat(torch.eye(3, device=focal_lengths.device), "... -> n ...", n=len(focal_lengths)).clone()
    intrinsics[:, 0, 0] = focal_lengths[:, 0]
    intrinsics[:, 1, 1] = focal_lengths[:, 1]
    intrinsics[:, 0, 2] = principal_points[:, 0]
    intrinsics[:, 1, 2] = principal_points[:, 1]
    return intrinsics


def fast_inverse(intrinsics: torch.Tensor) -> torch.Tensor:
    focal_length = get_focal_lengths(intrinsics)
    principal_point = get_principal_point(intrinsics)
    return as_matrix(1 / focal_length, -principal_point / focal_length)


def _normalize_shape_and_type(intrinsics: torch.Tensor | np.ndarray) -> torch.Tensor:
    if isinstance(intrinsics, np.ndarray):
        intrinsics = torch.from_numpy(intrinsics)
    if intrinsics.ndim == 2:
        intrinsics = intrinsics.unsqueeze(0)
    return intrinsics


def get_fov_deg(intrinsics: torch.Tensor | np.ndarray, image_size_wh: tuple) -> torch.Tensor:
    focal_lengths = get_focal_lengths(intrinsics)
    image_size_wh_tensor = torch.tensor(default_convert(image_size_wh)).float()
    if image_size_wh_tensor.ndim == 1:
        image_size_wh_tensor = image_size_wh_tensor.unsqueeze(0)
    fov = 2 * torch.atan2(image_size_wh_tensor / 2, focal_lengths)
    return torch.rad2deg(fov)


def get_focal_length_from_fov_deg(fov_deg: torch.Tensor, image_size_wh: tuple) -> torch.Tensor:
    if fov_deg.ndim == 1:
        fov_deg = fov_deg.unsqueeze(0)
    fov_rad = torch.deg2rad(fov_deg)
    image_size_wh_tensor = torch.tensor(default_convert(image_size_wh)).float()
    if image_size_wh_tensor.ndim == 1:
        image_size_wh_tensor = image_size_wh_tensor.unsqueeze(0)
    return image_size_wh_tensor / (2 * torch.tan(fov_rad / 2))


def scale(intrinsics: torch.Tensor | np.ndarray, scale_factor: float) -> torch.Tensor:
    intrinsics = _normalize_shape_and_type(intrinsics)
    intrinsics = intrinsics.clone()
    intrinsics[:, [0, 1], [0, 1]] *= scale_factor
    intrinsics[:, [0, 1], [2, 2]] *= scale_factor
    return intrinsics


def apply(intrinsics: torch.Tensor, points: torch.Tensor):
    assert points.shape[-1] == 3
    assert intrinsics.shape[-2:] == (3, 3)
    return (intrinsics @ points.unsqueeze(-1)).squeeze(-1)
