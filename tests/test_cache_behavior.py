import importlib.util
import json
import os
import sys
import tempfile
import types
import unittest
from pathlib import Path
from unittest import mock


REPO_ROOT = Path(__file__).resolve().parents[1]


def load_package():
    name = "comfyui_morphgs_test"
    for module_name in list(sys.modules):
        if module_name == name or module_name.startswith(name + "."):
            del sys.modules[module_name]
    spec = importlib.util.spec_from_file_location(
        name,
        REPO_ROOT / "__init__.py",
        submodule_search_locations=[str(REPO_ROOT)],
    )
    package = importlib.util.module_from_spec(spec)
    sys.modules[name] = package
    spec.loader.exec_module(package)
    return package, sys.modules[name + ".nodes"]


class CacheBehaviorTests(unittest.TestCase):
    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.root = Path(self.temp.name)
        self.input_dir = self.root / "input"
        self.models_dir = self.root / "models"
        self.input_dir.mkdir()
        (self.models_dir / "diffusion_models").mkdir(parents=True)
        (self.models_dir / "checkpoints").mkdir()

        self.folder_paths = types.ModuleType("folder_paths")
        self.folder_paths.models_dir = str(self.models_dir)
        self.folder_paths.folder_names_and_paths = {}
        self.folder_paths.get_input_directory = lambda: str(self.input_dir)
        self.folder_paths.get_folder_paths = lambda name: [str(self.models_dir / name)]
        self.folder_paths.get_filename_list = lambda _name: []
        self.folder_paths.get_full_path = lambda _name, filename: str(self.models_dir / "diffusion_models" / filename)
        self.folder_paths.get_output_directory = lambda: str(self.root / "output")
        sys.modules["folder_paths"] = self.folder_paths

        self.package, self.nodes = load_package()
        self.nodes.config.MORPHGS_HOME = str(self.root / "morphgs")

    def tearDown(self):
        sys.modules.pop("folder_paths", None)
        self.temp.cleanup()

    def test_diffusion_models_is_first_checkpoint_location(self):
        self.package._register_model_folders()
        paths, extensions = self.folder_paths.folder_names_and_paths["morphgs_sv4d_checkpoints"]
        self.assertEqual(paths[0], str(self.models_dir / "diffusion_models"))
        self.assertIn(".safetensors", extensions)

        checkpoint = self.models_dir / "diffusion_models" / "sv4d2.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        self.folder_paths.get_filename_list = lambda _name: ["sv4d2.safetensors"]
        self.assertEqual(self.nodes._sv4d_checkpoint_options(), ["sv4d2.safetensors"])
        options = self.nodes.MorphGSPreprocessVideo.INPUT_TYPES()["required"]["sv4d_mode"][1]
        self.assertNotIn("remote", options)

    def test_arap_joint_labels_follow_cached_sample_indices(self):
        source = (REPO_ROOT / "morphgs_src" / "src" / "model" / "MorphGS.py").read_text(encoding="utf-8")
        sampled = source.index("self.sampled_indices = _indicies")
        labels = source.index("self.rig.get_lbs_parts(self.sampled_indices)")
        model = source.index("ParametricModel(self.rig, gaussian_params, lbs_parts, dominant_joints")
        self.assertLess(sampled, labels)
        self.assertLess(labels, model)
        self.assertNotIn("self.rig.get_lbs_parts()", source)

    def test_character_cache_tracks_source_and_settings(self):
        source = self.input_dir / "character.glb"
        source.write_bytes(b"first")
        calls = []

        def fake_blender(_script, args, timeout=None):
            calls.append(("blender", args[2], args[3]))
            char_dir = Path(args[1])
            (char_dir / "rigging").mkdir(parents=True, exist_ok=True)
            (char_dir / "mesh.obj").write_text("mesh", encoding="utf-8")
            (char_dir / "rigging" / "mesh_ori_rig.txt").write_text("rig", encoding="utf-8")
            (char_dir / "rigging" / "conversion_meta.json").write_text(
                json.dumps({
                    "detected_height_m": 1.82,
                    "target_height": 1.82,
                    "height_decision": "auto: trusted imported file units",
                }),
                encoding="utf-8",
            )
            return "converted"

        def fake_python(_script, args, timeout=None, env=None):
            calls.append(("python", None))
            feature_dir = Path(args[0]) / "feature"
            feature_dir.mkdir(parents=True, exist_ok=True)
            (feature_dir / "features.pt").write_bytes(b"features")
            return "preprocessed"

        node = self.nodes.MorphGSPreprocessCharacter()
        input_types = node.INPUT_TYPES()["required"]
        self.assertNotIn("height_mode", input_types)
        self.assertNotIn("target_height", input_types)
        self.assertEqual(node.RETURN_NAMES[-1], "detected_height_m")
        with mock.patch.object(self.nodes, "run_blender_script", fake_blender), mock.patch.object(
            self.nodes, "run_python", fake_python
        ):
            result = node.run("character.glb", False)
            self.assertEqual(result[0], "character")
            self.assertAlmostEqual(result[2], 1.82)
            self.assertIn("detected 1.8200 m", result[1])
            self.assertEqual(calls[0][2], "auto_from_file_units")
            node.run("character.glb", False)
            self.assertEqual(len(calls), 2)

            before = node.IS_CHANGED("character.glb", False)
            source.write_bytes(b"changed source")
            after = node.IS_CHANGED("character.glb", False)
            self.assertNotEqual(before, after)
            node.run("character.glb", False)
            self.assertEqual(len(calls), 4)

            node.run("character.glb", True)
            self.assertEqual(len(calls), 6)

        self.assertEqual(self.nodes._stage_name_from_input("avatars/My Hero.glb", "character_name"), "My_Hero")

    def test_blender_converter_measures_evaluated_height_and_has_safe_fallback(self):
        source = (REPO_ROOT / "scripts" / "mesh_to_morphgs.py").read_text(encoding="utf-8")
        self.assertIn("evaluated_get(depsgraph)", source)
        self.assertIn("effective_target_height = detected_height_m", source)
        self.assertIn("effective_target_height = target_height", source)
        self.assertIn('"detected_height_m": detected_height_m', source)

    def test_upload_frontend_and_legacy_workflow_migration_are_bundled(self):
        frontend = (REPO_ROOT / "web" / "morphgs_inputs.js").read_text(encoding="utf-8")
        self.assertEqual(self.package.WEB_DIRECTORY, "./web")
        self.assertIn('"/upload/image"', frontend)
        self.assertIn('"Upload character"', frontend)
        self.assertIn('"Upload video"', frontend)
        self.assertIn("migrateLegacyWidgetValues", frontend)
        self.assertNotIn("file_upload", (REPO_ROOT / "nodes.py").read_text(encoding="utf-8"))

    def test_video_cache_tracks_max_frames_and_clears_old_frames(self):
        source = self.input_dir / "motion.mp4"
        source.write_bytes(b"video")
        checkpoint = self.models_dir / "diffusion_models" / "sv4d2.safetensors"
        checkpoint.write_bytes(b"checkpoint")
        calls = []

        def fake_python(script, args, timeout=None, env=None):
            calls.append((Path(script).name, tuple(args)))
            if Path(script).name == "mask_video.py":
                Path(args[1]).write_bytes(b"rgb")
            else:
                scene = Path(args[0]).parent.name
                color_dir = Path(self.nodes.config.MORPHGS_HOME) / "demo" / "processed_videos" / scene / "view_0" / "color"
                color_dir.mkdir(parents=True, exist_ok=True)
                (color_dir / "000.png").write_bytes(b"frame")
            return "ok"

        node = self.nodes.MorphGSPreprocessVideo()
        patches = (
            mock.patch.object(self.nodes, "run_python", fake_python),
            mock.patch.object(self.nodes, "_stage_sv4d_checkpoint", lambda *_args: "staged"),
            mock.patch.object(self.nodes, "_disable_xformers_in_sv4d_config", lambda *_args: "patched"),
            mock.patch.object(self.nodes, "_chunk_sgm_attention_batches", lambda: "patched"),
            mock.patch.object(self.nodes, "_align_vae_decode_dtype", lambda: "patched"),
            mock.patch.object(self.nodes, "_require_cuda_for_sv4d", lambda: "SV4D/DINO device: test GPU"),
        )
        with patches[0], patches[1], patches[2], patches[3], patches[4], patches[5]:
            node.run("motion.mp4", False, "sv4d2.safetensors", True, 12, False)
            node.run("motion.mp4", False, "sv4d2.safetensors", True, 12, False)
            self.assertEqual(len(calls), 2)
            self.assertNotEqual(
                node.IS_CHANGED("motion.mp4", False, "sv4d2.safetensors", True, 12, False),
                node.IS_CHANGED("motion.mp4", False, "sv4d2.safetensors", True, 24, False),
            )

            stale = Path(self.nodes.config.MORPHGS_HOME) / "demo" / "processed_videos" / "motion" / "view_0" / "color" / "999.png"
            stale.write_bytes(b"stale")
            node.run("motion.mp4", False, "sv4d2.safetensors", True, 24, False)
            self.assertEqual(len(calls), 4)
            self.assertFalse(stale.exists())
            self.assertIn("--sv4d_max_frames", calls[-1][1])
            self.assertIn(24, calls[-1][1])


if __name__ == "__main__":
    unittest.main()
