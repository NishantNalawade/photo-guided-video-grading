import subprocess
import tempfile
import unittest
from pathlib import Path

from streamlit.testing.v1 import AppTest

from video_processing import ffmpeg_path


class AppTests(unittest.TestCase):
    def app(self):
        app = AppTest.from_file(str(Path(__file__).with_name("app.py"))).run(timeout=60)
        self.assertFalse(app.exception)
        return app

    def test_device_diagnostics_and_upload_mode(self):
        app = self.app()
        devices = next(item for item in app.selectbox if item.label == "Processing device")
        self.assertIn(devices.value, ["CPU (multi-core)", "NVIDIA CUDA GPU"])
        source = next(item for item in app.radio if item.label == "Video source")
        self.assertEqual(source.value, "Local file (recommended for large videos)")
        self.assertTrue(any(item.label == "Local video path" for item in app.text_input))
        source.set_value("Browser upload").run(timeout=60)
        self.assertFalse(app.exception)
        self.assertFalse(app.text_input)

    def test_invalid_local_path_is_reported(self):
        app = self.app()
        app.text_input[0].set_value(str(Path(tempfile.gettempdir()) / "vcm_missing_test_video.mp4")).run(timeout=60)
        self.assertFalse(app.exception)
        self.assertTrue(any("does not exist" in item.value for item in app.error))

    def test_local_preview_export_and_result_persist_across_settings(self):
        with tempfile.TemporaryDirectory(prefix="vcm_ui_") as directory:
            source = Path(directory) / (Path(directory).name + ".mp4")
            subprocess.run([
                ffmpeg_path(), "-v", "error", "-y", "-f", "lavfi", "-i",
                "testsrc2=size=160x90:rate=12:duration=0.5",
                "-c:v", "libx264", str(source),
            ], check=True, capture_output=True)
            output = Path(__file__).parent / "exports" / f"{source.stem}_color_matched.mp4"
            try:
                app = self.app()
                app.text_input[0].set_value(str(source)).run(timeout=60)
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertTrue(any("Detected: 6 frames" in item.value for item in app.caption))
                next(item for item in app.button if item.label == "Process video and export MP4").click().run(timeout=60)
                self.assertFalse(app.exception)
                self.assertFalse(app.error)
                self.assertTrue(output.is_file())
                self.assertTrue(any("Export saved locally" in item.value for item in app.success))
                next(item for item in app.slider if item.label == "Contrast").set_value(1.1).run(timeout=60)
                self.assertFalse(app.exception)
                self.assertTrue(any("Export saved locally" in item.value for item in app.success))
            finally:
                output.unlink(missing_ok=True)


if __name__ == "__main__":
    unittest.main()
