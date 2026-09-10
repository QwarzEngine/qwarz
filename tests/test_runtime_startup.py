import json
import os
from pathlib import Path
from types import SimpleNamespace
import unittest
from unittest.mock import patch

from qwasar_runtime.engine import configure_gpu, verify_artifact


class StartupTests(unittest.TestCase):
    def test_artifact_hash_must_match_full_tree_manifest_pin(self):
        manifest = Path(__file__).resolve().parents[1] / "benchmarks/manifests/qwen38-27b-rtx5090-v1.json"
        expected = json.loads(manifest.read_text())["model"]["artifact_sha256"]
        with patch("qwasar_bench.environment.sha256_path", return_value=expected):
            self.assertEqual(verify_artifact(Path("/fake/model")), expected)
        with patch("qwasar_bench.environment.sha256_path", return_value="0" * 64):
            with self.assertRaisesRegex(ValueError, "artifact hash"):
                verify_artifact(Path("/fake/model"))

    def test_invalid_gpu_mapping_is_rejected_without_overwriting(self):
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "1"}):
            with self.assertRaises(ValueError):
                configure_gpu(SimpleNamespace())
            self.assertEqual(os.environ["CUDA_VISIBLE_DEVICES"], "1")

    def test_visible_device_must_be_single_rtx5090(self):
        for count, name in ((1, "NVIDIA RTX 4090"), (2, "NVIDIA GeForce RTX 5090")):
            cuda = SimpleNamespace(device_count=lambda: count, get_device_name=lambda index: name)
            with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
                with self.assertRaises(ValueError):
                    configure_gpu(SimpleNamespace(cuda=cuda))
        cuda = SimpleNamespace(device_count=lambda: 1, get_device_name=lambda index: "NVIDIA GeForce RTX 5090")
        with patch.dict(os.environ, {"CUDA_VISIBLE_DEVICES": "0"}):
            configure_gpu(SimpleNamespace(cuda=cuda))
