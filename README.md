# Video Color Matcher

A local Streamlit app for matching a video's overall color palette to one or more reference photos.

## Features
- Multiple JPG/PNG reference photos, selectable together or one at a time
- LAB color-distribution transfer with adjustable match strength, warmth/coolness, contrast and saturation
- First-frame before/after preview
- **Direct local-file input for large videos**, without uploading or copying the source
- **Video uploads up to 5 GB** (Streamlit upload limits are configured in `.streamlit/config.toml`); uploads are staged once per session, not on every slider change
- **Multi-core CPU processing** with a CPU thread control
- **Optional NVIDIA CUDA GPU grading** through CUDA-enabled PyTorch + Kornia
- Optional NVIDIA NVENC H.264 hardware encoding, independently selectable from grading, verified by an actual test encode
- Direct single-pass MP4 encoding (no intermediate video or second video encode), bounded frame batches and live throughput/ETA
- MP4 export with original audio where available
- Source video and references are processed locally; no hosted media service is used

## Requirements
- Windows, macOS or Linux
- Python 3.12 recommended
- Plenty of free disk space for source, temporary files and export

## Run on Windows
1. Install 64-bit Python 3.12 from https://www.python.org/downloads/ (enable the Python Launcher option).
2. Extract this folder.
3. Double-click `run_windows.bat`.
4. The app opens at `http://localhost:8501`.

The launcher creates/reuses a Python 3.12 virtual environment and installs the pinned base dependencies. The upload limit is 5 GB.

## Run on macOS / Linux
```bash
python3.12 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
pip install -r requirements.txt
python -m streamlit run app.py
```

## Optional NVIDIA GPU support
GPU mode requires an NVIDIA GPU with a compatible driver and a **CUDA-enabled PyTorch build**. The normal `requirements.txt` intentionally does not install the very large GPU runtime automatically.

On Windows, run the normal launcher once, close the app, then double-click `install_gpu_windows.bat`. This installs CUDA 12.8 PyTorch and Kornia into **this app's `.venv`**, then runs a CUDA color-transform self-test. The download is large and requires a compatible NVIDIA GPU and recent driver. Restart the app afterward.

For other platforms or a driver requiring a different CUDA build:

1. Activate the app's `.venv` (on Windows: `.venv\Scripts\activate.bat` in Command Prompt).
2. Open https://pytorch.org/get-started/locally/ and select the correct OS, pip and CUDA version for your system. Run the exact PyTorch install command it gives you, using this environment's `python -m pip`.
3. Then run:
   ```bash
   pip install -r requirements-gpu.txt
   ```
4. Restart the app. If CUDA is detected, select **NVIDIA CUDA GPU** in the Performance sidebar.

The sidebar shows either the tested GPU name or a specific dependency/driver error. CUDA grading errors are reported, not silently switched to CPU mid-export. CPU remains usable without these optional packages. AMD/Intel GPUs are not supported for CUDA grading.

NVENC requires a supported NVIDIA encoder and driver, not PyTorch. A bundled FFmpeg listing `h264_nvenc` does **not** mean the hardware works: the app tests a tiny encode before enabling this option. If unavailable, the sidebar explicitly shows CPU H.264 encoding and the diagnostic. If hardware encoding fails during export, the error is shown rather than silently degrading output quality.

## Large video notes
- Select **Local file (recommended for large videos)** and enter the full path on the computer running the app, e.g. `C:\Videos\input.mp4`. This avoids Streamlit upload RAM, browser transfer, and staging a second source copy. Local input is not subject to the browser's 5 GB upload limit.
- 5 GB is the maximum **uploaded source file size**, not a guarantee that every computer can process it. Browser uploads are still held in RAM and require disk space for a staged source. Staging is session-isolated and reused across settings changes.
- Only a bounded batch of decoded frames is processed at once. CPU worker count and CUDA batch size are limited by resolution/memory, avoiding nested OpenCV worker oversubscription. GPU reference tensors are reused throughout an export.
- Frames are piped directly into FFmpeg for H.264 encoding while grading, with original audio transcoded to AAC where present. There is no full-size intermediate encode. Allow disk space for the final output; an existing export is replaced only after a successful encode. Audio shorter than the video does not truncate the video.
- The app saves exports to the `exports` folder beside `app.py`. Very large outputs are not passed back through the browser download button; use the saved local file.
- Large source/output videos are not embedded in browser video players or download buttons (64 MiB threshold), preventing another large in-memory media copy. First-frame previews are capped at 960 pixels wide and source metadata/first frames are cached across settings changes.
- For reliable 5 GB workflows, prefer local input and choose **1080p max** or **720p max** if full resolution is not needed. Both width and height are constrained, including portrait videos; H.264 output dimensions are rounded down to even numbers.
- Duration and resolution, not compressed file size alone, determine processing cost. Live frames/sec and an estimated remaining time are shown. No fixed throughput is guaranteed.
- GPU mode batches color transforms on CUDA; OpenCV decoding still uses CPU. NVENC only accelerates encoding. CPU encoding uses the `veryfast` x264 preset at CRF 18.

## Validation
Run in the app environment:
```bash
python -m unittest -v test_video_processing test_app
```
Tests cover color-transfer compatibility, bounded batches, GPU diagnostics, upload reuse/session isolation, direct export with/without audio, full video retention when audio is shorter, failed-export cleanup, and UI preview/export/result persistence. Tensor color-parity/batching tests run when optional PyTorch and Kornia are installed; actual CUDA/NVENC operation additionally requires compatible hardware.

## Limitations
- This is an MVP, not a professional color-grading system. Whole-frame LAB statistics can affect skin tones and depend on scene/lighting similarity.
- Preview uses a resized first frame; export processes the full video at the selected export resolution. Small differences from resizing, CPU/OpenCV versus GPU/Kornia LAB conversion, and H.264 compression are expected.
- Export uses the source's reported frame rate (constant-rate output); variable-frame-rate timestamps, HDR metadata and subtitle tracks are not preserved.
- Start with a short test clip before processing a long 4K video.
