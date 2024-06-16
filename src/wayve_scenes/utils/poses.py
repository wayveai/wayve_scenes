from functools import singledispatch
from typing import Tuple
import torch
from pytorch3d.transforms import Transform3d as TorchTransform3D
from scipy.spatial.transform import Rotation as R
import plotly.graph_objects as go
from pytorch3d.transforms import Transform3d

import wayve_scenes.utils.transmat as transmat


def torch3d_pose_to_wayve(transform: TorchTransform3D) -> Tuple[torch.Tensor, torch.Tensor]:
    rotvec_torch3d = R.from_matrix(transmat.rotmat_col_major(transform).numpy()).as_rotvec()
    return (
        torch.from_numpy(R.from_rotvec(rotvec_torch3d[:, [2, 0, 1]]).as_matrix()),
        transmat.translation(transform)[:, [2, 0, 1]],
    )


@singledispatch
def wayve_pose_to_torch3d(rotation_matrix_col_major: torch.Tensor, translation: torch.Tensor):
    assert rotation_matrix_col_major.ndim == 3
    assert translation.ndim == 2

    return (
        TorchTransform3D()
        .rotate_axis_angle(90, "X")
        .rotate_axis_angle(90, "Z")
        .rotate(rotation_matrix_col_major.permute(0, 2, 1))
        .translate(*translation.T)
        .rotate_axis_angle(-90, "X")
        .rotate_axis_angle(-90, "Y")
    )


@wayve_pose_to_torch3d.register
def _(transform: TorchTransform3D):
    return wayve_pose_to_torch3d(*transmat.to_rotation_translation(transform))


def opencv_point_to_wayve_point(points_opencv: torch.Tensor) -> torch.Tensor:
    points_wayve = points_opencv.clone()
    points_wayve = points_wayve[..., [2, 0, 1]]
    points_wayve[..., 1:] *= -1
    return points_wayve


def opencv_pose_to_wayve_pose(g_rdf_any: torch.Tensor) -> torch.Tensor:
    g_flu_rdf = torch.tensor(
        [
            [0, 0, 1, 0],
            [-1, 0, 0, 0],
            [0, -1, 0, 0],
            [0, 0, 0, 1],
        ],
        dtype=torch.float,
        device=g_rdf_any.device,
    )
    return g_flu_rdf @ g_rdf_any



def wayve_pose_to_torch3d(rotation_matrix_col_major: torch.Tensor, translation: torch.Tensor):
    assert rotation_matrix_col_major.ndim == 3
    assert translation.ndim == 2

    return (
        TorchTransform3D()
        .rotate_axis_angle(90, "X")
        .rotate_axis_angle(90, "Z")
        .rotate(rotation_matrix_col_major.permute(0, 2, 1))
        .translate(*translation.T)
        .rotate_axis_angle(-90, "X")
        .rotate_axis_angle(-90, "Y")
    )


def set_layout(fig):
    axes = dict(visible=True, showbackground=False, showgrid=True, showline=True, showticklabels=True, autorange=True)
    fig.update_layout(
        template="plotly_dark",
        scene=dict(xaxis=axes, yaxis=axes, zaxis=axes, aspectmode="data", dragmode="orbit"),
        margin=dict(l=0, r=0, b=0, t=10, pad=0),
    )

