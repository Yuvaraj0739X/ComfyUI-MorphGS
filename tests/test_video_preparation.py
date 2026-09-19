import importlib.util
import tempfile
import unittest
from pathlib import Path
from unittest import mock

import numpy as np
from skimage.morphology import thin

ROOT = Path(__file__).resolve().parents[1]


def load_file(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


class VideoPreparationTests(unittest.TestCase):
    def test_gpu_backend_download_failure_keeps_existing_cpu_backend(self):
        installer = load_file("installer_test", ROOT / "install.py")
        with mock.patch.object(installer, "_python_subprocess", side_effect=[
            (0, ""), (0, "['CPUExecutionProvider']")
        ]), mock.patch.object(installer, "run", side_effect=RuntimeError("download unavailable")) as run, \
                mock.patch.object(installer, "pip_install") as install:
            with self.assertRaisesRegex(RuntimeError, "download unavailable"):
                installer.ensure_rembg_backend()
            self.assertEqual(run.call_count, 1)
            self.assertIn("download", run.call_args.args[0])
            install.assert_not_called()

    def test_live_subprocess_output_and_timeout(self):
        from test_cache_behavior import load_package
        _, nodes = load_package()
        import sys
        runner = sys.modules[nodes.__package__ + ".process_utils"]
        with tempfile.TemporaryDirectory() as directory:
            script = Path(directory) / "step.py"
            script.write_text("print('stage ready', flush=True)\n", encoding="utf-8")
            with mock.patch.object(runner.config, "MORPHGS_HOME", directory), \
                    mock.patch("builtins.print") as console:
                result = runner.run_python(str(script), [], timeout=5)
                self.assertIn("stage ready", result)
                self.assertTrue(any("stage ready" in str(c) for c in console.call_args_list))
                script.write_text("import time; time.sleep(10)\n", encoding="utf-8")
                with self.assertRaises(runner.subprocess.TimeoutExpired):
                    runner.run_python(str(script), [], timeout=0.1)

    def test_cropped_thinning_preserves_full_frame_result(self):
        helper = load_file("mask_utils_test", ROOT / "morphgs_src/src/utils/mask_utils.py")
        rng = np.random.default_rng(42)
        masks = [np.zeros((80, 90)), np.ones((80, 90))]
        for offset in (0, 20, 50):
            mask = np.zeros((80, 90), dtype=np.uint8)
            mask[offset:offset+30, offset:offset+30] = rng.integers(0, 2, (30, 30))
            masks.append(mask)
        mask = np.zeros((80, 90))
        mask[20:60, 30:70] = 1
        masks.append(mask)
        for mask in masks:
            np.testing.assert_array_equal(helper.thin_foreground(mask), thin(mask > 0) * 255)

    def test_decoder_limits_frames_before_masking(self):
        module = load_file("mask_video_test", ROOT / "scripts/mask_video.py")
        with tempfile.TemporaryDirectory() as directory, mock.patch.object(module.subprocess, "run") as run:
            module.extract_frames("source.mp4", directory, 12)
            args = run.call_args.args[0]
            self.assertEqual(args[args.index("-frames:v") + 1], "12")
            module.extract_frames("source.mp4", directory, 0)
            self.assertNotIn("-frames:v", run.call_args.args[0])


if __name__ == "__main__":
    unittest.main()
