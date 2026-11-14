import unittest

import torch

from wayve_scenes.utils.metrics import compute_ssim


class SSIMMaskTests(unittest.TestCase):
    def test_excluded_pixels_do_not_enter_valid_windows(self):
        generator = torch.Generator().manual_seed(123)
        target = torch.rand((31, 31, 3), generator=generator)
        mask = torch.zeros((31, 31), dtype=torch.bool)
        mask[10:21, 10:21] = True
        prediction = torch.where(mask[..., None], target, 1 - target)
        # The 11x11 window centered at (15, 15) contains only valid pixels.
        self.assertAlmostEqual(compute_ssim(prediction, target, mask).item(), 1.0, places=6)

    def test_mask_with_no_valid_window_returns_nan(self):
        target = torch.rand((31, 31, 3), generator=torch.Generator().manual_seed(2))
        mask = torch.zeros((31, 31), dtype=torch.bool)
        mask[15, 15] = True
        self.assertTrue(torch.isnan(compute_ssim(target, target, mask)).all())

    def test_all_valid_mask_preserves_identical_image_score(self):
        target = torch.rand((31, 31, 3), generator=torch.Generator().manual_seed(3))
        mask = torch.ones((31, 31), dtype=torch.bool)
        self.assertAlmostEqual(compute_ssim(target, target, mask).item(), 1.0, places=6)


if __name__ == '__main__':
    unittest.main()
