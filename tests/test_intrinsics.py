import unittest

import numpy as np
import torch

from wayve_scenes.utils import intrmat


class IntrinsicsPrecisionTests(unittest.TestCase):
    def test_double_precision_calibration_is_preserved(self):
        focal = torch.tensor([1000.123456789, 900.987654321], dtype=torch.float64)
        principal = torch.tensor([640.123456789, 360.987654321], dtype=torch.float64)
        matrix = intrmat.as_matrix(focal, principal)
        self.assertEqual(matrix.dtype, torch.float64)
        torch.testing.assert_close(matrix[0, [0, 1], [0, 1]], focal, rtol=0, atol=0)
        torch.testing.assert_close(matrix[0, :2, 2], principal, rtol=0, atol=0)

    def test_fast_inverse_keeps_double_precision(self):
        matrix = torch.tensor(
            [[[1000.123456789, 0, 640.123456789],
              [0, 900.987654321, 360.987654321],
              [0, 0, 1]]], dtype=torch.float64,
        )
        inverse = intrmat.fast_inverse(matrix)
        self.assertEqual(inverse.dtype, matrix.dtype)
        torch.testing.assert_close(matrix @ inverse, torch.eye(3, dtype=matrix.dtype)[None], rtol=0, atol=1e-12)

    def test_numpy_and_float32_inputs(self):
        matrix = intrmat.as_matrix(np.array([10, 20], dtype=np.float32), np.array([3, 4], dtype=np.float32))
        self.assertEqual(matrix.dtype, torch.float32)
        torch.testing.assert_close(matrix, torch.tensor([[[10., 0., 3.], [0., 20., 4.], [0., 0., 1.]]]))

    def test_mixed_precision_keeps_principal_point_precision(self):
        matrix = intrmat.as_matrix(torch.tensor([10., 20.]), torch.tensor([3.123456789, 4.], dtype=torch.float64))
        self.assertEqual(matrix.dtype, torch.float64)
        self.assertEqual(matrix[0, 0, 2].item(), 3.123456789)

    def test_integer_calibration_still_produces_floating_matrix(self):
        matrix = intrmat.as_matrix(torch.tensor([10, 20]), torch.tensor([3, 4]))
        self.assertTrue(matrix.is_floating_point())


if __name__ == '__main__':
    unittest.main()
