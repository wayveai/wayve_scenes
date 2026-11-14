import importlib.util
import json
from pathlib import Path
import sys
import tempfile
from types import ModuleType, SimpleNamespace
import unittest
from unittest.mock import Mock, patch

import numpy as np
from PIL import Image


def load_converter():
    # Isolate the external reconstruction readers and PLY writer. The converter,
    # image conversion, paths, and JSON serialization run without substitution.
    parsing = ModuleType('nerfstudio.data.utils.colmap_parsing_utils')
    parsing.qvec2rotmat = Mock(return_value=np.eye(3))
    parsing.read_cameras_binary = Mock(return_value={1: object()})
    parsing.read_images_binary = Mock()
    colmap = ModuleType('nerfstudio.process_data.colmap_utils')
    colmap.parse_colmap_camera_params = Mock(return_value={'camera_model': 'OPENCV'})
    colmap.create_ply_from_colmap = Mock()
    rich = ModuleType('nerfstudio.utils.rich_utils')
    rich.CONSOLE = Mock()
    modules = {module.__name__: module for module in (parsing, colmap, rich)}
    path = Path(__file__).resolve().parents[1] / 'src/wayve_scenes/utils/colmap_utils.py'
    spec = importlib.util.spec_from_file_location('_converter_under_test', path)
    converter = importlib.util.module_from_spec(spec)
    with patch.dict(sys.modules, modules):
        spec.loader.exec_module(converter)
    return converter, parsing


class ColmapMaskTests(unittest.TestCase):
    def setUp(self):
        self.converter, self.parsing = load_converter()
        self.directory = tempfile.TemporaryDirectory()
        self.addCleanup(self.directory.cleanup)
        self.root = Path(self.directory.name)

    def convert(self, filename, **kwargs):
        self.parsing.read_images_binary.return_value = {
            1: SimpleNamespace(qvec=np.array([1., 0., 0., 0.]), tvec=np.zeros(3),
                               name=filename, camera_id=1)
        }
        self.assertEqual(self.converter.colmap_to_json(self.root, self.root, **kwargs), 1)
        return json.loads((self.root / 'transforms.json').read_text())['frames'][0]

    def test_default_does_not_require_mask_files(self):
        frame = self.convert('front-forward/001.jpeg')
        self.assertNotIn('mask_path', frame)

    def test_false_does_not_require_mask_files(self):
        frame = self.convert('front-forward/001.jpeg', use_masks=False)
        self.assertNotIn('mask_path', frame)

    def test_image_extensions_map_to_png_masks(self):
        for extension in ('.jpeg', '.jpg', '.png', '.JPEG'):
            with self.subTest(extension=extension):
                mask = self.root / 'masks/front-forward/001.png'
                mask.parent.mkdir(parents=True, exist_ok=True)
                Image.new('RGB', (2, 2), color='white').save(mask)
                frame = self.convert('front-forward/001' + extension, use_masks=True)
                self.assertEqual(frame['mask_path'], 'masks/front-forward/001.png')
                with Image.open(mask) as converted:
                    self.assertEqual(converted.mode, 'L')


if __name__ == '__main__':
    unittest.main()
