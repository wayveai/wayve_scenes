from __future__ import annotations
from pathlib import Path
from typing import Any, Dict, List, Optional
import numpy as np
import pycolmap
import torch
import torch.jit
from scipy.spatial.transform import Rotation
import cv2
from tqdm import tqdm
import plotly.graph_objects as go

from pytorch3d.renderer.cameras import PerspectiveCameras
from pytorch3d.structures import Pointclouds
from pytorch3d.transforms import Transform3d as TorchTransform3D
from pytorch3d.vis.plotly_vis import plot_scene, get_camera_wireframe
from pytorch3d.renderer import FoVPerspectiveCameras

from wayve_scenes.utils.poses import opencv_point_to_wayve_point, opencv_pose_to_wayve_pose, set_layout
import wayve_scenes.utils.transmat as transmat 
import wayve_scenes.utils.intrmat as intrmat 


class WayveScene():
    g_world_vehicle: TorchTransform3D
    camera: PerspectiveCameras
    points_world: Pointclouds
    reconstruction: pycolmap.Reconstruction
    records: List[Dict[str, Any]]
    points_world: Pointclouds
    
    def __init__(self):
        self.g_world_vehicle = None
        self.camera = None
        self.points_world = None
        self.reconstruction = None
        self.records = None
        self.points_world = None
    
    
    def get_plot(self, prefix=""):
        plot = {}
        if self.points_world is not None:
            plot[prefix + "colmap_points"] = self.points_world
            
        if self.camera is not None:
            plot[prefix + "g_world_cam"] = self.camera
            
        return plot
    
        
    def visualise_colmap_scene(self, timestamp_index=0, marker_scale=0.2, pointcloud_max_points=50_000, extra_plots: Optional[Dict] = None):
        
        plot = self.get_plot()
        
        if extra_plots is not None:
            plot.update(extra_plots)
            
        
        # add cameras to the plot
        fig_3d_scene = plot_scene({"": plot}, camera_scale=marker_scale, pointcloud_max_points=pointcloud_max_points)
        
        set_layout(fig_3d_scene)
        
        camera_rotations = []
        camera_translations = []
        
        if self.records is not None:
            for record in self.records:
                pose = record["g_world_cam"]
                R, t = transmat.to_rotation_translation(pose)
                R = R.permute(0, 2, 1)
                camera_rotations.append(R)
                camera_translations.append(t)
                
        camera_rotations = torch.cat(camera_rotations, dim=0)
        camera_translations = torch.cat(camera_translations, dim=0)
                
        # Any instance of CamerasBase works, here we use FoVPerspectiveCameras
        cameras = FoVPerspectiveCameras(R=camera_rotations, T=camera_translations)
        
        camera_scale = 0.3
        cam_wires = get_camera_wireframe(camera_scale).to(cameras.device)
        cam_trans = cameras.get_world_to_view_transform()
        cam_wires_trans = cam_trans.transform_points(cam_wires).detach().cpu()

        if len(cam_wires_trans.shape) < 3:
            cam_wires_trans = cam_wires_trans.unsqueeze(0)

        nan_tensor = torch.Tensor([[float("NaN")] * 3])
        all_cam_wires = cam_wires_trans[0]
        for wire in cam_wires_trans[1:]:
            all_cam_wires = torch.cat((all_cam_wires, nan_tensor, wire))
        x, y, z = all_cam_wires.detach().cpu().numpy().T.astype(float)

        fig_3d_scene.add_trace(
            go.Scatter3d(x=x, y=y, z=z, marker={"size": 1}, name="camera_poses"),
        )
        
        
        # Remove all axis labels and ticks
        fig_3d_scene.update_layout(scene=dict(xaxis=dict(visible=False, title="", showticklabels=False), 
                                              yaxis=dict(visible=False, title="", showticklabels=False), 
                                              zaxis=dict(visible=False, title="", showticklabels=False)))
        
        
        fig_3d_scene.update_layout(
            scene=dict(
                camera=dict(
                    up=dict(x=0, y=0, z=1),
                    eye=dict(x=0, y=2, z=0),
                )
            ),
            margin=dict(t=30)  # left, right, top, bottom margins
        )
                     
        return fig_3d_scene


    def read_colmap_data(self, reconstruction_path: Path, load_points_camera: bool = False):
                
        reconstruction = pycolmap.Reconstruction(reconstruction_path)
        records = []
        
        for img in reconstruction.images.values():
            record = {}
            
            pose = img.cam_from_world
            tvec = torch.from_numpy(pose.translation).float()
            rot = torch.from_numpy(pose.rotation.quat)
            rot = torch.from_numpy(Rotation.from_quat(rot).as_matrix()).float()
            
            cam_props = reconstruction.cameras[img.camera_id]
            intrinsics = cam_props.calibration_matrix()
            intrinsics = torch.from_numpy(intrinsics).float()[None]

            # Colmap output is G_cam_world (projection from world to camera) - https://colmap.github.io/format.html#
            g_opencv_cam = transmat.fast_inverse(transmat.matrix_from_rotation_translation(rot, tvec))
            record["focal_length"] = intrmat.get_focal_lengths(intrinsics)[0]
            record["principal_point"] = intrmat.get_principal_point(intrinsics)[0]
            record["g_world_cam"] = opencv_pose_to_wayve_pose(g_rdf_any=g_opencv_cam[0])
            timestamp, position = int(Path(img.name).stem), Path(img.name).parent.name

            record["timestamp"] = int(timestamp)
            record["camera_position"] = position

            record["rgb"] = reconstruction_path.parent.parent / "images" / img.name
                        
            if load_points_camera:
                points_camera = get_points_camera_from_colmap_image(img, reconstruction)
                record["points_camera"] = Pointclouds(points_camera)

            records.append(record)
            
        # sort records according to timestamp, secundary according to camera_position
        records = sorted(records, key=lambda x: (x["timestamp"], x["camera_position"]))
        
        self.unique_timestamps = sorted(list(set([record["timestamp"] for record in records])))
            
        self.records = records

        bbs = reconstruction.compute_bounding_box(0.01, 0.99)
        pts3d = [
            (p3D.xyz, p3D.color)
            for _, p3D in reconstruction.points3D.items()
            if ((p3D.xyz >= bbs[0]).all() and (p3D.xyz <= bbs[1]).all() and p3D.error <= 6.0 and p3D.track.length() >= 2)
        ]
        
        if len(pts3d) > 0:
            points_opencv = torch.from_numpy(np.stack([p for p, _ in pts3d])).float()[None]
            points_wayve = opencv_point_to_wayve_point(points_opencv=points_opencv)

            colors = torch.from_numpy(np.stack([c for _, c in pts3d])).to(torch.uint8)[None]

            points_world = Pointclouds(points_wayve, features=colors / 255.0)
        else:
            points_world = None
            
        self.points_world = points_world
                
        
    def render_cameras_video(self, video_file: Path):
        
        print("Rendering camera video to: ", video_file)
        video_writer = cv2.VideoWriter(str(video_file), cv2.VideoWriter_fourcc(*"mp4v"), 10, (1920 * 3, 1080 * 2))
        
        image_positions = {
            "front-forward": (0, 1920),
            "left-forward": (0, 0),
            "right-forward": (0, 1920 * 2),
            "left-backward": (1080, 0),
            "right-backward": (1080, 1920 * 2),
        }
        
        # Create a video from the images
        video_canvas = np.zeros((2 * 1080, 3 * 1920, 3), dtype=np.uint8)
        num_cams = 5

        for record_idx, record in tqdm(enumerate(self.records), total=len(self.records)): 
                       
            imfile = str(record["rgb"])
            camera_name = Path(imfile).parent.name
            
            img = cv2.imread(imfile)
            
            image_pos = image_positions[camera_name]
            video_canvas[image_pos[0]:image_pos[0] + 1080, image_pos[1]:image_pos[1] + 1920] = img
        
            if (record_idx + 1) % num_cams == 0:
                video_writer.write(video_canvas)
                video_canvas = np.zeros((2 * 1080, 3 * 1920, 3), dtype=np.uint8)
                
        video_writer.release()
        print("Done rendering camera video")
            

def get_points_camera_from_colmap_image(img: pycolmap.Image, reconstruction: pycolmap.Reconstruction) -> torch.Tensor:
    p2ds = img.get_valid_points2D()
    if len(p2ds) == 0:
        return torch.zeros((1, 0, 3), dtype=torch.float32)

    xyz_world = np.array([reconstruction.points3D[p2d.point3D_id].xyz for p2d in p2ds])

    # COLMAP OpenCV convention: z is always positive
    xyz = (img.rotation_matrix() @ xyz_world.T) + img.tvec[:, None]

    # Mean reprojection error in image space
    errors = np.array([reconstruction.points3D[p2d.point3D_id].error for p2d in p2ds])

    # Number of frames in which each frame is visible
    n_visible = np.array([reconstruction.points3D[p2d.point3D_id].track.length() for p2d in p2ds])

    idx = np.where(
        (errors <= 2.0)  # Ignore points with reprojection greater than 2 px
        & (n_visible >= 3)  # Only consider points that are visible in more than 3 images
    )[0]
    cam_coords = torch.from_numpy(xyz[:, idx]).float()
    return cam_coords.T.unsqueeze(0)
