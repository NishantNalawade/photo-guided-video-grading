import math
import os
import shutil
import subprocess
import tempfile
import time
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from functools import lru_cache, partial
from pathlib import Path

import cv2
import numpy as np


@dataclass(frozen=True)
class VideoInfo:
    width: int
    height: int
    fps: float
    frames: int


@lru_cache(maxsize=1)
def cuda_stack():
    try:
        import torch
        import kornia
    except (ImportError, OSError) as exc:
        raise RuntimeError(
            "CUDA grading needs CUDA-enabled PyTorch and Kornia in this app's "
            f"Python environment. Run install_gpu_windows.bat. Details: {exc}"
        ) from exc
    if not torch.cuda.is_available():
        raise RuntimeError(
            f"PyTorch {torch.__version__} cannot access CUDA "
            f"(CUDA build: {torch.version.cuda}). Check the NVIDIA driver and "
            "install a CUDA-enabled PyTorch build in the app's environment."
        )
    try:
        device = torch.device("cuda")
        with torch.inference_mode():
            probe = torch.zeros((1, 3, 2, 2), device=device)
            kornia.color.lab_to_rgb(kornia.color.rgb_to_lab(probe)).sum().item()
    except RuntimeError as exc:
        raise RuntimeError(f"CUDA grading self-test failed: {exc}") from exc
    return torch, kornia, device


@lru_cache(maxsize=1)
def gpu_status():
    try:
        torch, _, _ = cuda_stack()
        return True, torch.cuda.get_device_name(0)
    except RuntimeError as exc:
        return False, str(exc)


def reference_stats(images, use_gpu=False):
    # One reference representation for both backends: OpenCV's 8-bit LAB.
    samples = []
    for rgb in images:
        if rgb is not None:
            thumb = cv2.resize(rgb, (256, 256), interpolation=cv2.INTER_AREA)
            samples.append(cv2.cvtColor(thumb, cv2.COLOR_RGB2LAB).reshape(-1, 3))
    if not samples:
        return None, None
    pixels = np.concatenate(samples).astype(np.float32)
    return pixels.mean(axis=0), pixels.std(axis=0).clip(1.0, None)


def transfer_lab(rgb, ref_mean, ref_std, strength):
    lab = cv2.cvtColor(rgb, cv2.COLOR_RGB2LAB)
    mean, std = cv2.meanStdDev(lab)
    matched = (lab.astype(np.float32) - mean.reshape(3).astype(np.float32))
    matched *= ref_std / std.reshape(3).clip(1.0, None).astype(np.float32)
    matched += ref_mean
    out = lab * (1.0 - strength) + matched * strength
    return cv2.cvtColor(np.clip(out, 0, 255).astype(np.uint8), cv2.COLOR_LAB2RGB)


def grade_cpu(rgb, ref_mean, ref_std, strength, warmth, contrast, saturation):
    out = transfer_lab(rgb, ref_mean, ref_std, strength) if ref_mean is not None and strength else rgb
    if warmth == 0 and contrast == 1 and saturation == 1:
        return out
    x = out.astype(np.float32)
    x[:, :, 0] += warmth * 0.45
    x[:, :, 2] -= warmth * 0.45
    x = np.clip((x - 127.5) * contrast + 127.5, 0, 255).astype(np.uint8)
    if saturation == 1:
        return x
    hsv = cv2.cvtColor(x, cv2.COLOR_RGB2HSV).astype(np.float32)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * saturation, 0, 255)
    return cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2RGB)


class GPUGrader:
    def __init__(self, ref_mean, ref_std, strength, warmth, contrast, saturation):
        self.torch, self.kornia, self.device = cuda_stack()
        self.strength, self.warmth = strength, warmth
        self.contrast, self.saturation = contrast, saturation
        self.ref_mean = self.ref_std = None
        if ref_mean is not None and strength:
            scale = np.array([100.0 / 255.0, 1.0, 1.0], dtype=np.float32)
            offset = np.array([0.0, 128.0, 128.0], dtype=np.float32)
            self.ref_mean = self.torch.as_tensor(
                ref_mean * scale - offset, device=self.device
            ).view(1, 3, 1, 1)
            self.ref_std = self.torch.as_tensor(
                ref_std * scale, device=self.device
            ).view(1, 3, 1, 1)

    def __call__(self, frames):
        torch, color = self.torch, self.kornia.color
        with torch.inference_mode():
            x = torch.from_numpy(np.stack(frames)).to(self.device)
            x = x.permute(0, 3, 1, 2).float().div_(255)
            if self.ref_mean is not None:
                lab = color.rgb_to_lab(x)
                mean = lab.mean(dim=(2, 3), keepdim=True)
                minimum = lab.new_tensor([100.0 / 255.0, 1.0, 1.0]).view(1, 3, 1, 1)
                std = lab.std(dim=(2, 3), unbiased=False, keepdim=True).maximum(minimum)
                matched = (lab - mean) * (self.ref_std / std) + self.ref_mean
                lab = lab * (1.0 - self.strength) + matched * self.strength
                lab[:, 0:1].clamp_(0, 100)
                lab[:, 1:3].clamp_(-128, 127)
                x = color.lab_to_rgb(lab).clamp_(0, 1)
            x[:, 0:1] += self.warmth * 0.45 / 255.0
            x[:, 2:3] -= self.warmth * 0.45 / 255.0
            x = ((x - 0.5) * self.contrast + 0.5).clamp_(0, 1)
            if self.saturation != 1:
                hsv = color.rgb_to_hsv(x)
                hsv[:, 1:2] = (hsv[:, 1:2] * self.saturation).clamp_(0, 1)
                x = color.hsv_to_rgb(hsv).clamp_(0, 1)
            return x.permute(0, 2, 3, 1).mul(255).round().byte().cpu().numpy()


def grade_gpu(rgb, ref_mean, ref_std, strength, warmth, contrast, saturation):
    return GPUGrader(ref_mean, ref_std, strength, warmth, contrast, saturation)([rgb])[0]


@lru_cache(maxsize=1)
def ffmpeg_path():
    try:
        import imageio_ffmpeg
    except ImportError:
        executable = shutil.which("ffmpeg")
        if not executable:
            raise RuntimeError("FFmpeg is missing. Install the base requirements.")
        return executable
    return imageio_ffmpeg.get_ffmpeg_exe()


@lru_cache(maxsize=4)
def nvenc_status(ffmpeg):
    # An encoder listing alone does not establish driver/device availability.
    try:
        result = subprocess.run(
            [ffmpeg, "-hide_banner", "-loglevel", "error", "-f", "lavfi",
             "-i", "color=size=64x64:rate=1", "-frames:v", "1",
             "-c:v", "h264_nvenc", "-preset", "p4", "-f", "null", "-"],
            capture_output=True, text=True, timeout=30,
        )
    except subprocess.TimeoutExpired:
        return False, "NVENC self-test timed out. Check the NVIDIA driver."
    return result.returncode == 0, result.stderr.strip()


def inspect_video(path):
    cap = cv2.VideoCapture(str(path))
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not open this video. Check its format and file permissions.")
        info = VideoInfo(
            int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            cap.get(cv2.CAP_PROP_FPS),
            int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        )
        ok, frame = cap.read()
        if not ok or info.width < 2 or info.height < 2 or not math.isfinite(info.fps) or info.fps <= 0:
            raise RuntimeError("Could not read valid video frames, dimensions or frame rate.")
        return info, cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    finally:
        cap.release()


def staged_upload(video, state):
    identity = (video.file_id, video.name, video.size)
    if state.get("upload_identity") != identity or not Path(state["upload_path"]).is_file():
        work = tempfile.TemporaryDirectory(prefix="vcm_upload_")
        path = Path(work.name) / ("source" + Path(video.name).suffix.lower())
        try:
            with path.open("wb") as target:
                buffer = video.getbuffer()
                try:
                    for start in range(0, len(buffer), 8 * 1024**2):
                        target.write(buffer[start:start + 8 * 1024**2])
                finally:
                    buffer.release()
        except OSError:
            work.cleanup()
            raise
        old = state.get("upload_directory")
        state["upload_directory"] = work
        state["upload_identity"] = identity
        state["upload_path"] = str(path)
        if old is not None:
            old.cleanup()
    return Path(state["upload_path"])


def export_dimensions(info, resolution):
    limits = {"Keep original": None, "1080p max": (1920, 1080), "720p max": (1280, 720)}
    limit = limits[resolution]
    scale = min(1.0, limit[0] / info.width, limit[1] / info.height) if limit else 1.0
    return max(2, int(info.width * scale) // 2 * 2), max(2, int(info.height * scale) // 2 * 2)


def batch_size(width, height, workers, use_gpu):
    if use_gpu:
        torch, _, _ = cuda_stack()
        free, _ = torch.cuda.mem_get_info()
        budget = min(512 * 1024**2, int(free * 0.15))
        return max(1, min(8, budget // (width * height * 256)))
    return max(1, min(workers, (256 * 1024**2) // (width * height * 96)))


def export_video(input_path, output_path, info, resolution, settings, use_gpu,
                 cpu_threads, use_nvenc=False, progress=None):
    if Path(input_path).resolve() == Path(output_path).resolve():
        raise ValueError("The export path must not overwrite the source video.")
    width, height = export_dimensions(info, resolution)
    size = batch_size(width, height, cpu_threads, use_gpu)
    workers = min(cpu_threads, size)
    # Avoid a worker pool multiplied by OpenCV's own internal thread pool.
    cv2.setNumThreads(1 if workers > 1 and not use_gpu else cpu_threads)
    gpu = GPUGrader(*settings) if use_gpu else None
    cpu = partial(grade_cpu, ref_mean=settings[0], ref_std=settings[1],
                  strength=settings[2], warmth=settings[3], contrast=settings[4],
                  saturation=settings[5])
    ffmpeg = ffmpeg_path()
    if use_nvenc:
        available, detail = nvenc_status(ffmpeg)
        if not available:
            raise RuntimeError(f"NVENC is unavailable: {detail}")
    codec = (["-c:v", "h264_nvenc", "-preset", "p4", "-cq", "18"] if use_nvenc else
             ["-c:v", "libx264", "-preset", "veryfast", "-crf", "18", "-threads", str(cpu_threads)])
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    fd, temporary = tempfile.mkstemp(prefix="vcm_export_", suffix=".mp4", dir=output_path.parent)
    os.close(fd)
    cap = cv2.VideoCapture(str(input_path))
    process = None
    pool = None
    done = 0
    last_update = 0.0
    try:
        if not cap.isOpened():
            raise RuntimeError("Could not reopen the source video.")
        with tempfile.TemporaryFile() as log:
            command = [
                ffmpeg, "-hide_banner", "-loglevel", "error", "-y",
                "-f", "rawvideo", "-pix_fmt", "rgb24", "-s:v", f"{width}x{height}",
                "-r", str(info.fps), "-i", "pipe:0", "-i", str(input_path),
                "-map", "0:v:0", "-map", "1:a?", *codec,
                "-pix_fmt", "yuv420p", "-c:a", "aac", "-b:a", "192k",
                "-movflags", "+faststart", temporary,
            ]
            process = subprocess.Popen(command, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL, stderr=log)
            if workers > 1 and not use_gpu:
                pool = ThreadPoolExecutor(max_workers=workers)
            try:
                while True:
                    batch = []
                    for _ in range(size):
                        ok, frame = cap.read()
                        if not ok:
                            break
                        if (frame.shape[1], frame.shape[0]) != (width, height):
                            frame = cv2.resize(frame, (width, height), interpolation=cv2.INTER_AREA)
                        batch.append(cv2.cvtColor(frame, cv2.COLOR_BGR2RGB))
                    if not batch:
                        break
                    processed = gpu(batch) if gpu else (pool.map(cpu, batch) if pool else map(cpu, batch))
                    for frame in processed:
                        process.stdin.write(np.ascontiguousarray(frame).tobytes())
                        done += 1
                    now = time.monotonic()
                    if progress and now - last_update >= 0.25:
                        progress(done, info.frames)
                        last_update = now
                process.stdin.close()
                try:
                    code = process.wait(timeout=300)
                except subprocess.TimeoutExpired as exc:
                    raise RuntimeError("FFmpeg did not finish finalizing the MP4 within five minutes.") from exc
            except (BrokenPipeError, OSError) as exc:
                if process.poll() is None:
                    process.terminate()
                process.wait(timeout=30)
                log.seek(max(0, log.seek(0, os.SEEK_END) - 3000))
                raise RuntimeError("FFmpeg encoding failed: " + log.read().decode(errors="replace")[-3000:]) from exc
            if code:
                log.seek(max(0, log.seek(0, os.SEEK_END) - 3000))
                raise RuntimeError("FFmpeg encoding failed: " + log.read().decode(errors="replace")[-3000:])
        if not done:
            raise RuntimeError("The video decoder returned no frames.")
        # Frame counts can be approximate, but a large deficit indicates truncation.
        if info.frames > 0 and done < info.frames - max(2, int(info.frames * 0.01)):
            raise RuntimeError(f"Video decoding stopped early ({done:,} of {info.frames:,} frames). Export not saved.")
        os.replace(temporary, output_path)
        return done
    finally:
        cap.release()
        if pool:
            pool.shutdown(wait=True)
        if process:
            if process.poll() is None:
                process.terminate()
                try:
                    process.wait(timeout=10)
                except subprocess.TimeoutExpired:
                    process.kill()
                    process.wait()
            if process.stdin and not process.stdin.closed:
                try:
                    process.stdin.close()
                except OSError:
                    pass  # The encoder error is reported by the caller above.
        Path(temporary).unlink(missing_ok=True)
