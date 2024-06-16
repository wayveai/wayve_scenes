import json
from pathlib import Path
from typing import Dict, Optional
from PIL import Image

import numpy as np
import torch

from nerfstudio.data.utils.colmap_parsing_utils import (
    qvec2rotmat,
    read_cameras_binary,
    read_images_binary,
)
from nerfstudio.process_data.colmap_utils import parse_colmap_camera_params, create_ply_from_colmap
from nerfstudio.utils.rich_utils import CONSOLE


def colmap_to_json(
    recon_dir: Path,
    output_dir: Path,
    use_masks: bool = False,
    image_id_to_depth_path: Optional[Dict[int, Path]] = None,
    ply_filename="sparse_pc.ply",
    keep_original_world_coordinate: bool = False,
    use_single_camera_mode: bool = True,
) -> int:
    """Converts COLMAP's cameras.bin and images.bin to a JSON file.

    Args:
        recon_dir: Path to the reconstruction directory, e.g. "sparse/0"
        output_dir: Path to the output directory.
        use_masks: If True, use masks.
        image_id_to_depth_path: When including sfm-based depth, embed these depth file paths in the exported json
        keep_original_world_coordinate: If True, no extra transform will be applied to world coordinate.
                    Colmap optimized world often have y direction of the first camera pointing towards down direction,
                    while nerfstudio world set z direction to be up direction for viewer.
    Returns:
        The number of registered images.
    """

    cam_id_to_camera = read_cameras_binary(recon_dir / "cameras.bin")
    im_id_to_image = read_images_binary(recon_dir / "images.bin")
    if set(cam_id_to_camera.keys()) != {1}:
        CONSOLE.print(f"[bold yellow]Warning: More than one camera is found in {recon_dir}")
        print(cam_id_to_camera)
        use_single_camera_mode = False 
        out = {} 
    else:  # one camera for all frames
        out = parse_colmap_camera_params(cam_id_to_camera[1])

    frames = []
    for im_id, im_data in im_id_to_image.items():
        rotation = qvec2rotmat(im_data.qvec)

        translation = im_data.tvec.reshape(3, 1)
        w2c = np.concatenate([rotation, translation], 1)
        w2c = np.concatenate([w2c, np.array([[0, 0, 0, 1]])], 0)
        c2w = np.linalg.inv(w2c)
        # Convert from COLMAP's camera coordinate system (OpenCV) to ours (OpenGL)
        c2w[0:3, 1:3] *= -1
        if not keep_original_world_coordinate:
            c2w = c2w[np.array([0, 2, 1, 3]), :]
            c2w[2, :] *= -1

        name = im_data.name
        name = Path(f"./images/{name}")

        frame = {
            "file_path": name.as_posix(),
            "transform_matrix": c2w.tolist(),
            "colmap_im_id": im_id,
        }
        if use_masks is not None:
            # replace .jpeg with .png and images/ with masks/
            mask_path = str(name).replace("images/", "masks/").replace(".jpeg", ".png")
            frame["mask_path"] = Path(mask_path).as_posix()
            
            # We have to convert the mask images to a binary mask (1-channel 8bit PNG)
            # This is done in the next step
            mask = Image.open(output_dir / frame["mask_path"])
            mask = mask.convert("L")
            mask.save(output_dir / frame["mask_path"])
            
        if image_id_to_depth_path is not None:
            depth_path = image_id_to_depth_path[im_id]
            frame["depth_file_path"] = str(depth_path.relative_to(depth_path.parent.parent))

        if not use_single_camera_mode:  # add the camera parameters for this frame
            frame.update(parse_colmap_camera_params(cam_id_to_camera[im_data.camera_id]))

        frames.append(frame)

    out["frames"] = frames

    applied_transform = None
    if not keep_original_world_coordinate:
        applied_transform = np.eye(4)[:3, :]
        applied_transform = applied_transform[np.array([0, 2, 1]), :]
        applied_transform[2, :] *= -1
        out["applied_transform"] = applied_transform.tolist()

    # create ply from colmap
    assert ply_filename.endswith(".ply"), f"ply_filename: {ply_filename} does not end with '.ply'"
    create_ply_from_colmap(
        ply_filename,
        recon_dir,
        output_dir,
        torch.from_numpy(applied_transform).float() if applied_transform is not None else None,
    )
    out["ply_file_path"] = ply_filename

    with open(output_dir / "transforms.json", "w", encoding="utf-8") as f:
        json.dump(out, f, indent=4)

    return len(frames)