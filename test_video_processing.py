import importlib.util
import subprocess
import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import cv2
import numpy as np

import video_processing as vp


SETTINGS = (None, None, 0.65, 0, 1.0, 1.0)


class ColorTests(unittest.TestCase):
    def setUp(self):
        self.frame = np.random.default_rng(42).integers(0, 256, (64, 96, 3), dtype=np.uint8)

    def test_identity_and_zero_reference_strength(self):
        np.testing.assert_array_equal(vp.grade_cpu(self.frame, *SETTINGS), self.frame)
        mean, std = vp.reference_stats([self.frame])
        np.testing.assert_array_equal(
            vp.grade_cpu(self.frame, mean, std, 0, 0, 1, 1), self.frame
        )

    def test_cpu_transfer_preserves_previous_algorithm(self):
        mean, std = vp.reference_stats([255 - self.frame])
        lab = cv2.cvtColor(self.frame, cv2.COLOR_RGB2LAB).astype(np.float32)
        pixels = lab.reshape(-1, 3)
        matched = (lab - pixels.mean(0)) * (std / pixels.std(0).clip(1, None)) + mean
        expected = cv2.cvtColor(
            np.clip(lab * 0.35 + matched * 0.65, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB
        )
        actual = vp.transfer_lab(self.frame, mean, std, 0.65)
        self.assertLess(np.abs(actual.astype(float) - expected).mean(), 0.1)

    def test_reference_representation_does_not_depend_on_device(self):
        cpu = vp.reference_stats([self.frame], use_gpu=False)
        gpu = vp.reference_stats([self.frame], use_gpu=True)
        for a, b in zip(cpu, gpu):
            np.testing.assert_array_equal(a, b)
        self.assertEqual(vp.reference_stats([]), (None, None))

    def test_uniform_frame_remains_finite(self):
        frame = np.full_like(self.frame, 128)
        mean, std = vp.reference_stats([frame])
        result = vp.grade_cpu(frame, mean, std, 1, 30, 1.2, 1.4)
        self.assertEqual(result.dtype, np.uint8)
        self.assertTrue(np.isfinite(result).all())

    def test_resolution_and_memory_limits(self):
        info = vp.VideoInfo(2160, 3840, 30, 100)
        self.assertEqual(vp.export_dimensions(info, "1080p max"), (606, 1080))
        self.assertEqual(vp.export_dimensions(vp.VideoInfo(321, 241, 30, 100), "Keep original"), (320, 240))
        self.assertEqual(vp.batch_size(3840, 2160, 32, False), 1)
        self.assertLessEqual(vp.batch_size(1920, 1080, 32, False), 2)
        with patch.object(vp, "cuda_stack") as stack:
            torch = SimpleNamespace(cuda=SimpleNamespace(mem_get_info=lambda: (2 * 1024**3, 4 * 1024**3)))
            stack.return_value = (torch, None, None)
            self.assertEqual(vp.batch_size(3840, 2160, 32, True), 1)

    def test_gpu_failure_is_explicit_not_cpu_fallback(self):
        with patch.object(vp, "cuda_stack", side_effect=RuntimeError("CUDA unavailable")):
            with self.assertRaisesRegex(RuntimeError, "CUDA unavailable"):
                vp.grade_gpu(self.frame, *SETTINGS)

    def test_nvenc_is_runtime_tested(self):
        vp.nvenc_status.cache_clear()
        failed = subprocess.CompletedProcess([], 1, "", "CUDA_ERROR_NO_DEVICE")
        with patch.object(vp.subprocess, "run", return_value=failed) as run:
            available, detail = vp.nvenc_status("test-ffmpeg")
            self.assertFalse(available)
            self.assertIn("CUDA_ERROR_NO_DEVICE", detail)
            self.assertIn("h264_nvenc", run.call_args.args[0])
            self.assertNotIn("-encoders", run.call_args.args[0])
        vp.nvenc_status.cache_clear()

    def test_upload_staged_once_and_isolated_by_session(self):
        from io import BytesIO
        data = BytesIO(b"test video")
        upload = SimpleNamespace(file_id="123", name="clip.mp4", size=10, getbuffer=data.getbuffer)
        a, b = {}, {}
        try:
            first = vp.staged_upload(upload, a)
            modified = first.stat().st_mtime_ns
            with patch.object(upload, "getbuffer", side_effect=AssertionError("Upload copied twice")):
                self.assertEqual(vp.staged_upload(upload, a), first)
            self.assertEqual(first.stat().st_mtime_ns, modified)
            other = vp.staged_upload(upload, b)
            self.assertNotEqual(first, other)
            upload.file_id = "replacement"
            replacement = vp.staged_upload(upload, a)
            self.assertFalse(first.exists())
            self.assertEqual(replacement.read_bytes(), b"test video")
            self.assertEqual(other.read_bytes(), b"test video")
        finally:
            for state in (a, b):
                if "upload_directory" in state:
                    state["upload_directory"].cleanup()

    @unittest.skipUnless(importlib.util.find_spec("torch") and importlib.util.find_spec("kornia"),
                         "Optional PyTorch/Kornia are not installed")
    def test_gpu_tensor_pipeline_batch_independence_and_color_parity(self):
        import torch
        import kornia
        mean, std = vp.reference_stats([255 - self.frame])
        settings = (mean, std, 0.65, 7, 1.1, 1.2)
        # Run the actual tensor implementation even on CI machines without CUDA.
        with patch.object(vp, "cuda_stack", return_value=(torch, kornia, torch.device("cpu"))):
            grader = vp.GPUGrader(*settings)
            frames = [self.frame, np.flip(self.frame, axis=0).copy()]
            batch = grader(frames)
            for i, frame in enumerate(frames):
                np.testing.assert_allclose(batch[i], grader([frame])[0], atol=1)
            cpu = vp.grade_cpu(self.frame, *settings)
            self.assertLess(np.abs(batch[0].astype(float) - cpu).mean(), 5)


class ExportTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.work = tempfile.TemporaryDirectory(prefix="vcm_tests_")
        cls.root = Path(cls.work.name)
        cls.ffmpeg = vp.ffmpeg_path()
        cls.source = cls.root / "source with audio.mp4"
        subprocess.run([
            cls.ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
            "-f", "lavfi", "-i", "testsrc2=size=160x90:rate=12:duration=1",
            "-f", "lavfi", "-i", "sine=frequency=440:duration=0.5",
            "-c:v", "libx264", "-pix_fmt", "yuv420p", "-c:a", "aac", str(cls.source),
        ], check=True, capture_output=True)

    @classmethod
    def tearDownClass(cls):
        cls.work.cleanup()

    def test_streamed_export_retains_frames_and_short_audio(self):
        info, _ = vp.inspect_video(self.source)
        output = self.root / "result.mp4"
        updates = []
        with patch.object(vp.subprocess, "Popen", wraps=subprocess.Popen) as encoder:
            done = vp.export_video(self.source, output, info, "Keep original", SETTINGS,
                                   False, 2, progress=lambda d, t: updates.append((d, t)))
            self.assertEqual(encoder.call_count, 1)
            command = encoder.call_args.args[0]
            self.assertIn("rawvideo", command)
            self.assertIn("pipe:0", command)
            self.assertNotIn("mp4v", command)
        result_info, _ = vp.inspect_video(output)
        self.assertEqual(done, 12)
        self.assertEqual(result_info.frames, 12)
        self.assertEqual((result_info.width, result_info.height), (160, 90))
        self.assertAlmostEqual(result_info.fps, 12)
        self.assertTrue(updates)
        audio = subprocess.run([
            self.ffmpeg, "-v", "error", "-i", str(output), "-map", "0:a:0",
            "-f", "null", "-",
        ], capture_output=True)
        self.assertEqual(audio.returncode, 0, audio.stderr.decode())
        self.assertFalse(list(self.root.glob("vcm_export_*")))

    def test_silent_source_exports_without_audio(self):
        source = self.root / "silent.mp4"
        subprocess.run([self.ffmpeg, "-v", "error", "-y", "-i", str(self.source),
                        "-an", "-c:v", "copy", str(source)], check=True, capture_output=True)
        info, _ = vp.inspect_video(source)
        output = self.root / "silent_result.mp4"
        self.assertEqual(vp.export_video(source, output, info, "720p max", SETTINGS, False, 1), 12)
        self.assertEqual(vp.inspect_video(output)[0].frames, 12)

    def test_failed_export_preserves_existing_output_and_cleans_temporary(self):
        info, _ = vp.inspect_video(self.source)
        output = self.root / "existing.mp4"
        output.write_bytes(b"existing export")
        with patch.object(vp, "grade_cpu", side_effect=RuntimeError("grading failed")):
            with self.assertRaisesRegex(RuntimeError, "grading failed"):
                vp.export_video(self.source, output, info, "Keep original", SETTINGS, False, 2)
        self.assertEqual(output.read_bytes(), b"existing export")
        self.assertFalse(list(self.root.glob("vcm_export_*")))

    def test_source_cannot_be_overwritten(self):
        info, _ = vp.inspect_video(self.source)
        with self.assertRaisesRegex(ValueError, "overwrite"):
            vp.export_video(self.source, self.source, info, "Keep original", SETTINGS, False, 1)

    def test_decode_failure_does_not_report_success(self):
        with self.assertRaisesRegex(RuntimeError, "Could not open"):
            vp.inspect_video(self.root / "missing.mp4")
        info, _ = vp.inspect_video(self.source)
        incorrect = vp.VideoInfo(info.width, info.height, info.fps, 120)
        output = self.root / "truncated.mp4"
        with self.assertRaisesRegex(RuntimeError, "stopped early"):
            vp.export_video(self.source, output, incorrect, "Keep original", SETTINGS, False, 1)
        self.assertFalse(output.exists())


if __name__ == "__main__":
    unittest.main()
