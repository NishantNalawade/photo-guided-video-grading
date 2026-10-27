import os
import time
from pathlib import Path

import cv2
import numpy as np
import streamlit as st

from video_processing import (
    export_video, ffmpeg_path, gpu_status, grade_cpu, grade_gpu,
    inspect_video, nvenc_status, reference_stats, staged_upload,
)

st.set_page_config(page_title="Video Color Matcher", page_icon="🎬", layout="wide")
st.title("🎬 Video Color Matcher")
st.caption("Match a video's overall color palette to reference photos. Processing stays on this computer.")


@st.cache_data(max_entries=24, show_spinner=False)
def read_image(data):
    bgr = cv2.imdecode(np.frombuffer(data, dtype=np.uint8), cv2.IMREAD_COLOR)
    if bgr is None:
        raise ValueError("Could not decode a reference photo. Please choose a valid JPG or PNG.")
    return cv2.cvtColor(bgr, cv2.COLOR_BGR2RGB)


@st.cache_data(max_entries=4, show_spinner=False)
def first_frame(path, modified, size):
    return inspect_video(path)


with st.sidebar:
    st.header("Color controls")
    strength = st.slider("Reference match strength", 0.0, 1.0, 0.65, 0.05)
    warmth = st.slider("Warm ↔ cool", -50, 50, 0, 1)
    contrast = st.slider("Contrast", 0.60, 1.60, 1.0, 0.05)
    saturation = st.slider("Saturation", 0.0, 2.0, 1.0, 0.05)
    resolution = st.selectbox("Export resolution", ["Keep original", "1080p max", "720p max"])
    st.divider()
    st.subheader("Performance")
    cpu_count = os.cpu_count() or 4
    cpu_threads = st.slider("CPU threads", 1, min(32, cpu_count), min(8, cpu_count),
                            help="Frame workers are limited by resolution to bound memory use. Encoding also uses this thread limit.")
    gpu_available, gpu_detail = gpu_status()
    gpu_mode = st.selectbox("Processing device", ["CPU (multi-core)", "NVIDIA CUDA GPU"],
                            index=1 if gpu_available else 0, disabled=not gpu_available)
    if gpu_available:
        st.success(f"CUDA grading ready: {gpu_detail}")
    else:
        st.warning(gpu_detail)
        st.caption("The normal launcher installs CPU dependencies only. GPU setup is a separate step; restart the app after installing.")
    use_gpu = gpu_mode == "NVIDIA CUDA GPU" and gpu_available
    try:
        nvenc_available, nvenc_detail = nvenc_status(ffmpeg_path())
    except (RuntimeError, OSError, TimeoutError) as exc:
        nvenc_available, nvenc_detail = False, str(exc)
    use_nvenc = st.checkbox("Use NVIDIA hardware encoding", value=nvenc_available, disabled=not nvenc_available,
                            help="Encoding is independent of CUDA color grading. Availability is checked with a real test encode.")
    if not nvenc_available:
        st.caption("Encoding: CPU H.264 (veryfast).")
        with st.expander("Hardware encoder diagnostics"):
            st.code(nvenc_detail or "NVENC is not available.")
    st.caption("For large videos, use a local file and 1080p or 720p export when full resolution is not needed.")

cv2.setNumThreads(cpu_threads)
grade_fn = grade_gpu if use_gpu else grade_cpu

st.subheader("1. Reference photos")
upload_mode = st.radio("How would you like to add reference photos?",
                       ["Select multiple photos at once", "Add photos one at a time"], horizontal=True)
refs_upload = []
if upload_mode == "Select multiple photos at once":
    refs_upload = st.file_uploader("Choose multiple JPG/PNG photos", type=["jpg", "jpeg", "png"],
                                   accept_multiple_files=True, key="reference_photos_multi") or []
else:
    photo_count = st.number_input("Number of reference photos", min_value=1, max_value=12, value=3, step=1)
    for i in range(int(photo_count)):
        uploaded = st.file_uploader(f"Reference photo {i + 1}", type=["jpg", "jpeg", "png"],
                                    key=f"reference_photo_single_{i}")
        if uploaded is not None:
            refs_upload.append(uploaded)
try:
    refs = [read_image(f.getvalue()) for f in refs_upload]
except ValueError as exc:
    st.error(str(exc))
    st.stop()
ref_mean, ref_std = reference_stats(refs)
if refs:
    st.caption(f"{len(refs)} reference photo(s) selected")
    cols = st.columns(min(4, len(refs)))
    for col, im, uploaded in zip(cols, refs, refs_upload):
        with col:
            st.image(im, caption=uploaded.name, use_container_width=True)
else:
    st.info("Add reference photos to guide the color palette. Without them, warmth, contrast and saturation still work.")

st.subheader("2. Video")
source_mode = st.radio("Video source", ["Local file (recommended for large videos)", "Browser upload"], horizontal=True)
source = None
try:
    if source_mode == "Local file (recommended for large videos)":
        st.caption("Enter a path on the computer running this app. The file is read directly, without an upload or source copy.")
        local_path = st.text_input("Local video path", placeholder=r"C:\Videos\input.mp4")
        if local_path.strip():
            source = Path(local_path.strip().strip('"')).expanduser().resolve()
            if not source.is_file():
                raise ValueError("The local video file does not exist or is not accessible.")
    else:
        st.caption("Maximum upload size: 5 GB. Browser uploads are held in RAM; prefer a local file for large videos.")
        video = st.file_uploader("Choose a video file (up to 5 GB)", type=["mp4", "mov", "mkv", "avi", "webm"],
                                 key="video_upload")
        if video is not None:
            source = staged_upload(video, st.session_state)
    if source is not None:
        stat = source.stat()
        info, rgb = first_frame(str(source), stat.st_mtime_ns, stat.st_size)
        st.caption(f"Source: {stat.st_size / 1024**3:.2f} GB")
        # Streamlit's video widget loads local media into memory; avoid it for large files.
        if stat.st_size < 64 * 1024**2:
            st.video(str(source))
        thumb = rgb
        if rgb.shape[1] > 960:
            thumb = cv2.resize(rgb, (960, max(1, round(rgb.shape[0] * 960 / rgb.shape[1]))),
                               interpolation=cv2.INTER_AREA)
        settings = (ref_mean, ref_std, strength, warmth, contrast, saturation)
        preview = grade_fn(thumb, *settings)
        a, b = st.columns(2)
        with a:
            st.image(thumb, caption="Original first frame", use_container_width=True)
        with b:
            st.image(preview, caption="Graded preview", use_container_width=True)
        duration = info.frames / info.fps
        st.caption(f"Detected: {info.frames:,} frames · {info.fps:.2f} FPS · approximately {duration / 60:.1f} minutes")
        if st.button("Process video and export MP4", type="primary", use_container_width=True):
            output_dir = Path(__file__).resolve().parent / "exports"
            output_path = output_dir / f"{source.stem}_color_matched.mp4"
            if source_mode == "Browser upload":
                output_path = output_dir / f"{Path(video.name).stem}_color_matched.mp4"
            if output_path.resolve() == source:
                raise ValueError("Export would overwrite the source. Rename or move the source before processing.")
            progress = st.progress(0.0, text="Preparing direct MP4 encode...")
            status = st.empty()
            started = time.monotonic()

            def update(done, total):
                elapsed = max(time.monotonic() - started, 0.001)
                fraction = min(done / total, 0.99) if total > 0 else 0.0
                progress.progress(fraction, text=f"Grading and encoding: {done:,} / {total:,} frames")
                rate = done / elapsed
                remaining = max(0, total - done) / rate if rate else 0
                status.info(f"{rate:.1f} frames/sec · estimated {remaining / 60:.1f} minutes remaining")

            frames = export_video(source, output_path, info, resolution, settings, use_gpu,
                                  cpu_threads, use_nvenc=use_nvenc, progress=update)
            elapsed = time.monotonic() - started
            progress.progress(1.0, text=f"Finished: {frames:,} frames in {elapsed:.1f} seconds")
            status.empty()
            st.session_state.last_export = str(output_path.resolve())
    else:
        st.caption("Tip: start with a short clip while tuning the look.")
except (OSError, ValueError, RuntimeError) as exc:
    st.error(str(exc))

saved = st.session_state.get("last_export")
if saved and Path(saved).is_file():
    output = Path(saved)
    st.caption("Last successful export. Settings changes do not reprocess automatically.")
    st.success(f"Export saved locally: {output}")
    if output.stat().st_size < 64 * 1024**2:
        st.video(str(output))
        st.download_button("Download graded video", data=output.read_bytes(), file_name=output.name,
                           mime="video/mp4", use_container_width=True)
    else:
        st.caption("Open the saved MP4 from the exports folder. Large exports are not loaded into browser/download memory.")
