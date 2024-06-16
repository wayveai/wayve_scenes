from typing import Optional
from PIL import Image
import torch
import numpy as np

from torchmetrics.functional.image.ssim import _ssim_update
from torchmetrics.image.fid import FrechetInceptionDistance
from torchmetrics.image.lpip import LearnedPerceptualImagePatchSimilarity


def get_metrics(image_pred_file: str, image_target_file: str, mask_file: str = None) -> dict:
    
    """
    Compute the PSNR, SSIM, and LPIPS between the predicted and target images.
    """
    
    # check that we indeed have different image files
    # assert image_pred_file != image_target_file, "The predicted and target images must be different."

    pred = np.array(Image.open(image_pred_file))
    target = np.array(Image.open(image_target_file))
    
    if mask_file is not None:
        mask = np.array(Image.open(mask_file).convert("L")).astype(bool)
        pred = pred * mask[..., None]
        target = target * mask[..., None]
    
    # remove alpha channel if it exists
    if pred.shape[-1] == 4:
        pred = pred[:, :, :3]
    if target.shape[-1] == 4:
        target = target[:, :, :3]    
            
    # explicit conversion torch float tensor in format (H, W, C) and normalize to [0, 1]
    pred = torch.tensor(pred).float() / 255
    target = torch.tensor(target).float() / 255
    
    return {
        "psnr": compute_psnr(pred, target, mask=None).item(),
        "ssim": compute_ssim(pred, target, mask=None).item(),
        "lpips": compute_lpips(pred, target, mask=None).item(),
    }


def get_fid_metric(image_pred_files_list: list, image_target_files_list: list) -> torch.Tensor:
    """
    Compute the FID between the predicted and target images in the given image file lists.
    """
    
    pred_images_stack = torch.stack([torch.tensor(np.array(Image.open(image_pred_file))).float() / 255 for image_pred_file in image_pred_files_list])
    target_images_stack = torch.stack([torch.tensor(np.array(Image.open(image_target_file))).float() / 255 for image_target_file in image_target_files_list])
    
    return compute_fid(pred_images_stack, target_images_stack, mask=None).item()


def compute_psnr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    normalise: Optional[bool] = False,
    do_filtering: Optional[bool] = False,
) -> torch.Tensor:
    """Compute the PSNR between the predicted and target images. Only the masked region is considered.

    Args:
        pred: The predicted image. Expected shape: (..., 3).
        target: The target image. Expected shape: (..., 3).
        mask: The mask of the region to compute the PSNR for. Expected shape: (...).

    Returns:
        The PSNR of the masked region of the image or the whole image if mask is None.
    """

    assert pred.shape == target.shape, "The predicted and target images must have the same shape."
    assert pred.dtype == target.dtype, "The predicted and target images must have the same data type."
    assert pred.device == target.device, "The predicted and target images must be on the same device."
    assert pred.shape[-1] == target.shape[-1] == 3, "The predicted/target image must have 3 channels."

    if mask is not None:
        assert mask.shape[:] == pred.shape[:-1], "The mask and predicted images must have the same spatial dimensions."
        assert mask.dtype == torch.bool, "The mask must be a boolean tensor."

    # Clamp the predicted and target image element values to the range [0, 1].
    pred = torch.clamp(pred, min=0.0, max=1.0)
    target = torch.clamp(target, min=0.0, max=1.0)

    if normalise:
        # normalise the target and pred to have the same average brightness
        mean = torch.mean(target)
        std = torch.std(target)

        target = (target - mean) / max(std, 1e-6)
        pred = (pred - mean) / max(std, 1e-6)

        # Scale the images
        target = (target - target.min()) / (target.max() - target.min())
        pred = (pred - pred.min()) / (pred.max() - pred.min())

    if mask is not None:
        mask = mask.to(pred.device)

        if do_filtering:
            # Filter the images, if we use the new way of filtering
            pred = pred[mask]
            target = target[mask]

            # check if pred and target still have data (if mask is all False, then pred and target will be empty tensors)
            if pred.shape[0] == 0:
                return torch.tensor([0.0]).fill_(float("nan"))

        else:
            # Blacken the images, if we use the old way of filtering
            pred[mask == 0] = 0
            target[mask == 0] = 0

            # check if pred and target still have non-black pixels (otherwise, return 0)
            if target.max() == 0:
                return torch.tensor([0.0]).fill_(float("nan"))

    # Compute the mean squared error.
    mse = torch.mean((pred - target) ** 2)

    # Compute the PSNR.
    psnr = 20 * torch.log10(1.0 / torch.sqrt(mse))
    
    # If the PSNR is infinite, set it to max value of 100.
    psnr[torch.isinf(psnr)] = 100.0

    return psnr


def compute_ssim(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """
    Compute the SSIM between the predicted and target images. Only the masked region is considered.

    Args:
        pred: The predicted image. Expected shape: (H, W, 3).
        target: The target image. Expected shape: (H, W, 3).
        mask: The mask of the region to compute the PSNR for. Expected shape: (H, W).

    Returns:
        torch.Tensor: The SSIM of the masked region of the image or the whole image if mask is None.
    """

    assert pred.shape == target.shape, "The predicted and target images must have the same shape."
    assert pred.shape[-1] == 3, "The predicted/target image must have 3 channels."
    assert len(pred.shape) == 3, "The images must have the shape (H, W, 3)."

    if mask is not None:
        assert mask.shape[:] == pred.shape[:-1], "The mask and predicted images must have the same spatial dimensions."
        assert mask.dtype == torch.bool, "The mask must be a boolean tensor."

    # If the images are given in the (H, W, 3) format, we need to convert them to the (1, 3, H, W) format for SSIM.
    pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
    target = target.unsqueeze(0).permute(0, 3, 1, 2)

    pred = torch.clamp(pred, min=0.0, max=1.0)
    target = torch.clamp(target, min=0.0, max=1.0)

    # Get the SSIM map.
    kernel_size = 11
    ssim_avg, sim_map = _ssim_update(pred, target, kernel_size=kernel_size, return_full_image=True)

    if mask is None:
        return ssim_avg[0]

    # Erode the mask by kernel_size//2
    def erode_mask(mask):
        mask = mask.view(-1, 1, *mask.shape[-2:])
        mask = mask.float()
        mask = torch.nn.functional.pad(
            mask, (kernel_size // 2, kernel_size // 2, kernel_size // 2, kernel_size // 2), mode="constant", value=0
        )
        mask = torch.nn.functional.max_pool2d(mask, kernel_size, stride=1)
        return mask > 0

    mask_eroded = erode_mask(mask)
    mask_eroded = mask_eroded.repeat(1, 3, 1, 1)

    sim_map_masked = sim_map[mask_eroded]

    # check if sim_map_masked still has data (if mask is all False, then sim_map_masked will be an empty tensor)
    if sim_map_masked.numel() == 0:
        return torch.tensor([0.0]).fill_(float("nan"))

    ssim_avg_masked = sim_map_masked.mean()

    return ssim_avg_masked


def compute_lpips(pred: torch.Tensor, target: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
    """Compute the LPIPS between a predicted and a target image.

    Args:
        pred: The predicted image. Expected shape: (H, W, 3).
        target: The target image. Expected hape: (H, W, 3).
        mask: The mask of the region to compute the LPIPS for. Shape: (H, W).

    Returns:
        The LPIPS of the two images.
    """

    assert pred.shape == target.shape, "The predicted and target images must have the same shape."
    assert pred.dtype == target.dtype, "The predicted and target images must have the same data type."
    assert pred.device == target.device, "The predicted and target images must be on the same device."
    assert pred.shape[-1] == 3, "The predicted/target image must have 3 channels."
    assert len(pred.shape) == 3, "The images must have the shape (H, W, 3)."

    pred = torch.clamp(pred, min=0.0, max=1.0)
    target = torch.clamp(target, min=0.0, max=1.0)

    if mask is not None:
        # We blacken the images outside the mask. This is necessary because LPIPS does not support masks.
        assert mask.shape[:] == pred.shape[:-1], "The mask and predicted images must have the same spatial dimensions."
        assert mask.dtype == torch.bool, "The mask must be a boolean tensor."
        mask = mask.to(pred.device)
        pred = pred * mask.unsqueeze(-1)
        target = target * mask.unsqueeze(-1)

        # check if pred and target still have non-black pixels (otherwise, return 0)
        if target.max() == 0:
            return torch.tensor([0.0]).fill_(float("nan"))

    # If the images are given in the (H, W, 3) format, we need to convert them to the (1, 3, H, W) format for SSIM.
    pred = pred.unsqueeze(0).permute(0, 3, 1, 2)
    target = target.unsqueeze(0).permute(0, 3, 1, 2)

    lpips = LearnedPerceptualImagePatchSimilarity(normalize=True).to(pred.device)

    return lpips(pred, target)


def compute_snr(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
    normalise: Optional[bool] = False,
    do_filtering: Optional[bool] = False,
) -> torch.Tensor:
    
    """Compute the SNR (not PSNR!) between the predicted and target images. Only the masked region is considered.

    Args:
        pred: The predicted image. Shape: (..., 3).
        target: The target image. Shape: (..., 3).
        mask: The mask of the region to compute the PSNR for. Shape: (...).

    Returns:
        The SNR of the masked region of the image or the whole image if mask is None.
    """

    assert pred.shape == target.shape, "The predicted and target images must have the same shape."
    assert pred.dtype == target.dtype, "The predicted and target images must have the same data type."
    assert pred.device == target.device, "The predicted and target images must be on the same device."
    assert pred.shape[-1] == target.shape[-1] == 3, "The predicted/target image must have 3 channels."

    if mask is not None:
        assert mask.shape[:] == pred.shape[:-1], "The mask and predicted images must have the same spatial dimensions."
        assert mask.dtype == torch.bool, "The mask must be a boolean tensor."

    # Clamp the predicted and target image element values to the range [0, 1].
    pred = torch.clamp(pred, min=0.0, max=1.0)
    target = torch.clamp(target, min=0.0, max=1.0)

    if normalise:
        # normalise the target and pred to have the same average brightness
        mean = torch.mean(target)
        std = torch.std(target)

        target = (target - mean) / max(std, 1e-6)
        pred = (pred - mean) / max(std, 1e-6)

        # Scale the images
        target = (target - target.min()) / (target.max() - target.min())
        pred = (pred - pred.min()) / (pred.max() - pred.min())

    if mask is not None:
        mask = mask.to(pred.device)

        if do_filtering:
            # Filter the images, if we use the new way of filtering
            pred = pred[mask]
            target = target[mask]

            # check if pred and target still have data (if mask is all False, then pred and target will be empty tensors)
            if pred.shape[0] == 0:
                return torch.tensor([0.0]).fill_(float("nan"))

        else:
            # Blacken the images, if we use the old way of filtering
            pred[mask == 0] = 0
            target[mask == 0] = 0

            # check if pred and target still have non-black pixels (otherwise, return NaN)
            if target.max() == 0:
                return torch.tensor([0.0]).fill_(float("nan"))

    # Compute the noise.
    noise = target - pred

    eps = torch.finfo(pred.dtype).eps

    # Compute the SNR
    snr_value = (torch.sum(target**2) + eps) / (torch.sum(noise**2) + eps)

    return 10 * torch.log10(snr_value)


def compute_fid(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: Optional[torch.Tensor] = None,
) -> torch.Tensor:
    
    """
    Compute the FID between the predicted and target images.
    """

    assert pred.shape == target.shape, "The predicted and target images must have the same shape."
    assert pred.dtype == target.dtype, "The predicted and target images must have the same data type."
    assert pred.device == target.device, "The predicted and target images must be on the same device."
    assert pred.dtype == torch.float32, "The images must be of type float32."
    assert pred.shape[-1] == 3, "The predicted/target image must have 3 channels."

    # If we pass in two [H, W, 3] images, expand their dims
    if len(pred.shape) == 3:
        pred = pred.unsqueeze(0)
        target = target.unsqueeze(0)
        mask = mask.unsqueeze(0) if mask is not None else None
    # else, we expect two [N, H, W, 3] images
    else:
        assert len(pred.shape) == 4, "The images must have the shape (N, H, W, 3)."

    pred = torch.clamp(pred, min=0.0, max=1.0)
    target = torch.clamp(target, min=0.0, max=1.0)

    if mask is not None:
        # We blacken the images outside the mask. This is necessary because LPIPS does not support masks.
        assert mask.shape[:] == pred.shape[:-1], "The mask and predicted images must have the same spatial dimensions."
        assert mask.dtype == torch.bool, "The mask must be a boolean tensor."
        mask = mask.to(pred.device)
        pred = pred * mask.unsqueeze(-1)
        target = target * mask.unsqueeze(-1)

        # check if pred and target still have non-black pixels (otherwise, return 0)
        if target.max() == 0:
            return torch.tensor([0.0]).fill_(float("nan"))

    # If the images are given in the (N, H, W, 3) format, we need to convert them to the (N, 3, H, W) format for SSIM.
    pred = pred.permute(0, 3, 1, 2)
    target = target.permute(0, 3, 1, 2)

    # If we only passed in 1 pair of images,
    # We need to repeat the images along the batch dimension to compute the FID (we need image distribution N > 1)
    if pred.shape[0] == 1:
        pred = pred.repeat(2, 1, 1, 1)
        target = target.repeat(2, 1, 1, 1)
        
    # convert pred and target to uint8 for FID
    pred = (pred * 255).to(torch.uint8)
    target = (target * 255).to(torch.uint8)
    
    fid = FrechetInceptionDistance(feature=64).to(pred.device)
    fid.update(target, real=True)
    fid.update(pred, real=False)
    
    fid_value = fid.compute()
    fid_value = torch.clamp(fid_value, min=0.0)

    return fid_value