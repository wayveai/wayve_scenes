from __future__ import annotations

from functools import singledispatch
from typing import Literal, Tuple

import numpy as np
import torch
from pytorch3d.transforms import Transform3d as TorchTransform3D, matrix_to_quaternion, quaternion_to_matrix


def to_transform_matrix_col_major(transform: TorchTransform3D) -> torch.Tensor:
    assert isinstance(transform, TorchTransform3D)
    return transform.get_matrix().permute(0, 2, 1)


def to_rotation_translation(transform: TorchTransform3D) -> Tuple[torch.Tensor, torch.Tensor]:
    return rotmat_col_major(transform), translation(transform)


def from_transform_matrix_col_major(transform: torch.Tensor) -> TorchTransform3D:
    assert isinstance(transform, torch.Tensor)
    return TorchTransform3D(matrix=transform.permute(0, 2, 1))


def from_rotation_translation(rot_col_major: torch.Tensor, trans: torch.Tensor) -> TorchTransform3D:
    return from_transform_matrix_col_major(matrix_from_rotation_translation(rot_col_major, trans))


def matrix_from_rotation_translation(rot_col_major: torch.Tensor, trans: torch.Tensor) -> torch.Tensor:
    if rot_col_major.ndim == 2:
        rot_col_major = rot_col_major[None]
    if trans.ndim == 1:
        trans = trans[None]

    mat = torch.eye(4, dtype=trans.dtype, device=trans.device).view(1, 4, 4).repeat(len(trans), 1, 1)
    mat[:, :3, 3] = trans
    mat[:, :3, :3] = rot_col_major
    return mat


@singledispatch
def translation(transform: TorchTransform3D) -> torch.Tensor:
    return translation(to_transform_matrix_col_major(transform))


@translation.register
def _(transform_matrix_col_major: torch.Tensor) -> torch.Tensor:
    if transform_matrix_col_major.ndim == 2:
        transform_matrix_col_major = transform_matrix_col_major[None]
    return transform_matrix_col_major[:, 0:3, 3]


@singledispatch
def rotmat_col_major(transform: TorchTransform3D) -> torch.Tensor:
    return rotmat_col_major(to_transform_matrix_col_major(transform))


@rotmat_col_major.register
def _(transform_matrix_col_major: torch.Tensor) -> torch.Tensor:
    if transform_matrix_col_major.ndim == 2:
        transform_matrix_col_major = transform_matrix_col_major[None]
    return transform_matrix_col_major[:, :3, :3]


def _get_axis(axis: int, rmat_col_major: torch.Tensor) -> torch.Tensor:
    return rmat_col_major[:, :, axis]


@singledispatch
def x_axis(transform: TorchTransform3D) -> torch.Tensor:
    return _get_axis(0, rotmat_col_major(transform))


@x_axis.register
def _(transform_matrix_col_major: torch.Tensor) -> torch.Tensor:
    return _get_axis(0, rotmat_col_major(transform_matrix_col_major))


@singledispatch
def y_axis(transform: TorchTransform3D) -> torch.Tensor:
    return _get_axis(1, rotmat_col_major(transform))


@y_axis.register
def _(transform_matrix_col_major: torch.Tensor) -> torch.Tensor:
    return _get_axis(1, rotmat_col_major(transform_matrix_col_major))


@singledispatch
def z_axis(transform: TorchTransform3D) -> torch.Tensor:
    return _get_axis(2, rotmat_col_major(transform))


@z_axis.register
def _(transform_matrix_col_major: torch.Tensor) -> torch.Tensor:
    return _get_axis(2, rotmat_col_major(transform_matrix_col_major))


@singledispatch
def fast_inverse(transform: TorchTransform3D) -> TorchTransform3D:
    return from_transform_matrix_col_major(fast_inverse(to_transform_matrix_col_major(transform)))


@fast_inverse.register
def _(transform_matrix_col_major: torch.Tensor) -> torch.Tensor:
    rot_inv = rotmat_col_major(transform_matrix_col_major).permute(0, 2, 1)
    trans = translation(transform_matrix_col_major)
    return matrix_from_rotation_translation(rot_inv, (rot_inv @ -trans.unsqueeze(-1)).view_as(trans))


def identity(batch=1) -> torch.Tensor:
    return torch.eye(4).view(1, 4, 4).repeat(batch, 1, 1)


def quat_from_matrix(rot_col_major: torch.Tensor) -> torch.Tensor:
    if rot_col_major.ndim == 2:
        rot_col_major = rot_col_major[None]
    return matrix_to_quaternion(rot_col_major.permute(0, 2, 1))


def quat_to_matrix(quat: torch.Tensor) -> torch.Tensor:
    if quat.ndim == 1:
        quat = quat[None]
    return quaternion_to_matrix(quat).permute(0, 2, 1)


def rotate(
    pose: torch.Tensor | TorchTransform3D, rot_matrix_col_major: torch.Tensor
) -> torch.Tensor | TorchTransform3D:
    """Rotate the pose in place (without changing the translation)."""
    pose_rot = rotmat_col_major(pose)
    pose_rot = pose_rot @ rot_matrix_col_major
    mat = matrix_from_rotation_translation(pose_rot, translation(pose))
    if isinstance(pose, torch.Tensor):
        return mat
    return from_transform_matrix_col_major(mat)


def apply(transform: torch.Tensor, points: torch.Tensor) -> torch.Tensor:
    assert points.ndim == transform.ndim - 1
    return (transform[..., :3, :3] @ points.unsqueeze(-1)).squeeze(-1) + transform[..., :3, 3]


def pose_is_close(pose1: torch.Tensor, pose2: torch.Tensor, dist_eps: float = 0.01, rot_deg_eps: float = 0.01) -> bool:
    dist = torch.norm(translation(pose1) - translation(pose2), dim=-1)
    if not np.allclose(dist, 0, atol=dist_eps):
        return False

    qvec1 = quat_from_matrix(rotmat_col_major(pose1))
    qvec2 = quat_from_matrix(rotmat_col_major(pose2))
    qvec1 = qvec1 / torch.norm(qvec1, dim=-1, keepdim=True)  # shape: (n, 4)
    qvec2 = qvec2 / torch.norm(qvec2, dim=-1, keepdim=True)  # shape: (n, 4)
    # Angle between two unit quaternions.
    # References:
    # https://petercorke.github.io/spatialmath-python/_modules/spatialmath/base/quaternions.html#qangle
    d = 2.0 * torch.atan2(torch.norm(qvec1 - qvec2, dim=-1), torch.norm(qvec1 + qvec2, dim=-1))
    return np.allclose(d, 0, atol=np.deg2rad(rot_deg_eps))


def rotation_reduce(rotmats: torch.Tensor, how: Literal["mean", "median"]) -> torch.Tensor:
    """
    Compute the mean/median rotation matrix from a set of rotation matrices.
    """
    assert rotmats.ndim == 3
    assert rotmats.shape[1:] == (3, 3)

    # Convert rotation matrices to quaternions.
    quats = quat_from_matrix(rotmats)
    assert quats.shape[1:] == (4,)

    # Compute the reduced quaternion.
    if how == "mean":
        quat_reduce = torch.mean(quats, dim=0)
    else:
        quat_reduce = torch.median(quats, dim=0).values
    quat_reduce = quat_reduce / torch.norm(quat_reduce)

    # Convert the mean quaternion to a rotation matrix.
    return quat_to_matrix(quat_reduce)


def reduce(transforms: torch.Tensor, how: Literal["mean", "median"]) -> torch.Tensor:
    """
    Reduce a set of transforms with a mean or a median.
    """
    assert transforms.ndim == 3
    assert transforms.shape[1:] == (4, 4)

    rotmats = rotmat_col_major(transforms)
    translations = translation(transforms)

    rotmat_reduce = rotation_reduce(rotmats, how)
    if how == "mean":
        translation_reduce = torch.mean(translations, dim=0)
    else:
        # Not super correct, because the median is taken for each column independently.
        # But should work for our purposes.
        translation_reduce = torch.median(translations, dim=0).values

    return matrix_from_rotation_translation(rotmat_reduce, translation_reduce)


def is_valid(transforms: torch.Tensor) -> bool:
    """
    Check that the transforms are valid.
    """
    if transforms.ndim == 2:
        transforms = transforms[None]
    assert transforms.shape[1:] == (4, 4)
    valid = True

    # Check that the rotation matrices are valid.
    rotmats = rotmat_col_major(transforms)
    # Check that the rotation matrices are orthogonal.
    valid = valid and torch.allclose(
        rotmats @ rotmats.transpose(-1, -2), torch.eye(3, device=rotmats.device)[None], atol=1e-06
    )
    # Check that the rotation matrices have nonnegative determinant.
    valid = valid and torch.all(torch.det(rotmats) >= torch.zeros(rotmats.shape[0], device=rotmats.device))

    # Check the bottom row of the tranform matrices.
    valid = valid and torch.all(transforms[..., 3, :] == torch.tensor([0.0, 0.0, 0.0, 1.0], device=transforms.device))
    return valid


def fix_rotation(rot_col_major: torch.Tensor) -> torch.Tensor:
    """
    Returns the nearest rotation matrix.
    Source: https://www.cs.cornell.edu/courses/cs6210/2022fa/lec/2022-10-06.pdf, section 7
    """
    u, s, v = torch.svd(rot_col_major)
    new_s = torch.ones_like(s).scatter(
        dim=-1, index=torch.argmin(s, dim=-1, keepdim=True), src=(u @ v.transpose(-1, -2)).det().sign()[..., None]
    )
    return u @ torch.diag_embed(new_s) @ v.transpose(-1, -2)


def fix(transforms: torch.Tensor):
    """
    Ensure the rotation matrix axes are orthogonal.
    """
    rotmats = rotmat_col_major(transforms)
    rotmats = fix_rotation(rotmats)
    return matrix_from_rotation_translation(rotmats, translation(transforms))


def fix_if_invalid(transforms: torch.Tensor) -> torch.Tensor:
    transform_shape = transforms.shape
    if not is_valid(transforms):
        transforms = fix(transforms)
        assert is_valid(transforms)
    return transforms.view(transform_shape)
