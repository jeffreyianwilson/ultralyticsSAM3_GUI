# ultralyticsSAM3_GUI

`ultralyticsSAM3_GUI` is a desktop-first SAM 3 application built on top of Ultralytics SAM 3. It provides a compact PySide6 GUI for image, folder, and video workflows, plus a reusable Python backend and CLI for automation and scripting.

The project is structured so the GUI, CLI, and backend all use the same `SAM3Ultralytics` runtime layer.

<img width="2554" height="1391" alt="image" src="https://github.com/user-attachments/assets/026f126a-f7e0-413a-8778-cd0af315ebbb" />

## Credits

Created by **Jeffrey Ian Wilson** with implementation support from **OpenAI Codex**.

## Disclaimer

This application was entirely written by AI, it is strongly recommended to review source code before usage.  All attempts to parse code for security concerns using OpenAI Codex have been made.  Use at your own risk.

## Features

- Ultralytics SAM 3 backend wrapper with a stable Python API
- PySide6 desktop GUI optimized for laptop workflows
- Text, point, box, and mask prompting for inference
- Manual mask authoring as a separate non-inference layer
- Image, folder, and video support
- Per-frame timeline controls with frame jump input
- Frame-scoped view filtering by class, ID, and instance
- Overlay, JSON, and raw mask export
- Cache-backed mask and result storage
- Reusable loaded model instance across runs for lower overhead

## Current Issues

- Exemplar prompting not fully supported.
- Video workflow not thoroughly tested.
## Project Layout

- `sam3_ultralytics/backend.py`: high-level backend API
- `sam3_ultralytics/gui.py`: GUI module entry point
- `sam3_ultralytics/gui_app.py`: main application window and workflow orchestration
- `sam3_ultralytics/gui_widgets.py`: preview canvas and custom widgets
- `sam3_ultralytics/cache_store.py`: disk-backed cache helpers
- `sam3_ultralytics/export.py`: export pipeline
- `sam3_ultralytics/cli.py`: command-line interface

## Installation

Use Python `3.11+`.

Install into your environment:

```bash
pip install -e .
```

Runtime dependencies declared in `pyproject.toml`:

- `numpy`
- `opencv-python`
- `Pillow`
- `PySide6`
- `timm`
- `ultralytics`
- `clip` from the Ultralytics CLIP repository

## Download the SAM 3 Checkpoint

This app expects a SAM 3 checkpoint file such as `sam3.pt`.

The official Meta SAM 3 repository points users to the gated Hugging Face model page for checkpoint access:

- Meta SAM 3 repo: [facebookresearch/sam3](https://github.com/facebookresearch/sam3)
- Official checkpoint page: [facebook/sam3 on Hugging Face](https://huggingface.co/facebook/sam3)

Typical flow:

1. Request access on the Hugging Face `facebook/sam3` page.
2. Sign in with the Hugging Face CLI if needed:

```bash
huggingface-cli login
```

3. Download `sam3.pt` from the model page, or place the checkpoint where you want to keep local models, for example:

```text
D:\cache\models\sam3.pt
```

4. In the GUI, use the `Model` selector to point the app at that checkpoint.

If you already have `sam3.pt`, you do not need to download it again.

## Launch

Run the GUI:

```bash
python -m sam3_ultralytics.gui
```

Installed entry points:

```powershell
sam3-gui
sam3-ultralytics
```

## GUI Workflow

### 1. Configure Runtime

- Select the SAM 3 checkpoint in `Model`
- Select `Device`
- Choose a `Cache` directory for disk-backed inference state
- Enable `Use compact cache archives (recommended)` to store masks in compact v2 archives instead of large raster cache files
- Use `Clear Cache` when you want to reset the cache directory

### 2. Load a Source

- `Open Image` for a single image
- `Open Folder` for an image sequence
- `Open Video` for video input

### 3. Configure Inference

Inside the `Inference` rollout:

- `Run Scope`
  - `Current Image / Frame`
  - `Entire Folder / Video`
- `Confidence`
- `Downscale before inference`
- `Scale Factor`
- `Text Prompt`
- `Load Mask` for prompt-mask inference
- `Prompt Class`
- `Prompt ID`
- `Append inferred masks`
- `Point Tool`
- `Box Tool`

Prompt tools are staged first. They do not trigger inference automatically. Press `Run` to execute inference.

### 4. Manual Masks

Inside the `Manual Masks` rollout:

- Manual masks are always metadata-only overlays
- They are not sent into SAM 3 inference
- They always use:
  - `ID = 0`
  - `Class = manualMask`

Supported manual mask actions:

- `Manual Mask Tool`
- `Shift + Left Mouse` to paint/add
- `Ctrl + Left Mouse` to erase/remove
- `Copy Mask`
- `Paste To Current`
- `Copy To All Frames`
- `Copy Range` with `Prev` and `Next` frame counts
- `Clear Manual Mask`

### 5. View and Export

Inside the `View/Export` rollout:

- overlay opacity
- label visibility
- mask visibility
- track ID visibility
- class, ID, and instance filters
- filtered export (exports only masks visible in the current view filters)

- `Export Directory`
- `Auto-export masks after inference`
- `Export merged masks only`
- `Invert exported masks`
- `Mask Dilation (px)`
- `Export Masks`

## Sequence and Video Behavior

### Folder Runs

Folder runs are processed one item at a time through the GUI:

1. move the view to the current image
2. run inference for that image
3. load the result into the view
4. advance to the next image

Folder runs are independent per-image inference. They do not use hidden tracking across the folder.
Use the playback row to navigate with `Prev`, `Play`, `Next`, slider scrub, or direct frame jump by entering a frame number.

### Video Runs

Video runs support:

- current-frame inference
- full-sequence runs
- video tracking when mask/interactive initialization requires it

## Prompt Types

Supported prompt paths in the current app:

- text prompts
- point prompts
- box prompts
- mask prompts
- mixed text + interactive refinement when supported by the backend compatibility layer

The app does not currently expose exemplar prompting in the GUI.

## Python API

```python
from sam3_ultralytics import SAM3Ultralytics

backend = SAM3Ultralytics(r"D:\cache\models\sam3.pt", device="cuda:0")
backend.load()

result = backend.predict_image(
    r"D:\path\to\image.jpg",
    text_prompt="person",
    mask_input=r"D:\path\to\rough_mask.png",
    mask_id=1,
    mask_label="person",
)

print(result.source)
print(result.mode)
print(result.timings)
```

Folder prediction:

```python
results = backend.predict_image_sequence(
    [r"D:\images\frame_0001.jpg", r"D:\images\frame_0002.jpg"],
    text_prompt="person",
)
```

Video frame prediction:

```python
results = backend.predict_video_frames(
    r"D:\clips\clip.mp4",
    frame_indices=[0, 1, 2],
    text_prompt="person",
)
```

## CLI

Examples:

Image:

```powershell
sam3-ultralytics image --model "D:\cache\models\sam3.pt" --device cuda:0 --text person "D:\images\image.jpg"
```

Directory:

```powershell
sam3-ultralytics image --model "D:\cache\models\sam3.pt" --text person --all-items --inference-scale 0.5 "D:\images"
```

Video tracking:

```powershell
sam3-ultralytics video-track --model "D:\cache\models\sam3.pt" --mask "D:\masks\frame0.png" "D:\clips\clip.mp4"
```

Mixed batch:

```powershell
sam3-ultralytics batch --model "D:\cache\models\sam3.pt" --text person --inference-scale 0.5 "D:\images" "D:\clips\clip.mp4"
```

## Caching and Memory Behavior

The GUI supports a disk-backed cache directory.

What is cached:

- prompt masks
- manual masks
- inference result masks
- Ultralytics writable settings/config under the cache directory
- versioned cache data under the selected cache root in a `v2` namespace

When compact cache archives are enabled, result masks are stored as cropped, bit-packed per-frame archives instead of full-frame raster `.npy` files.

When inference downscale is enabled, SAM 3 runs on the reduced frame size for speed and memory savings, while exports and overlays are restored to the original source dimensions.

This keeps large mask arrays off memory across longer sessions and makes repeated GUI use more stable on large inputs.

## Export Outputs

Image outputs use deterministic naming such as:

- `<source_stem>_mask_001.png`
- `<source_stem>_track_012.png`
- `<source_stem>_merged_mask.png`
- `<source_stem>_overlay.png`
- `<source_stem>_results.json`

Video outputs use a predictable frame-oriented structure such as:

- `frames/frame_000001_overlay.png`
- `masks/frame_000001_obj_001.png`
- `masks/frame_000001_track_012.png`
- `json/frame_000001.json`
- `annotated_video.mp4`

Manual masks are included in merged mask export.

## Troubleshooting

### GUI does not launch

Make sure `PySide6` is installed in the same environment used to start the app.

### SAM 3 runtime import errors

Install missing runtime dependencies from `pyproject.toml`.

### CUDA not being used

Check the active environment:

```powershell
C:\Users\jeffr\.conda\envs\ultralytics\python.exe -c "import torch; print(torch.cuda.is_available())"
```

### Reset the app cache

Use `Clear Cache` in the GUI, or delete the configured cache directory manually.

## Changelog

### 2026-03-27

- Added playback frame jump input (`Frame #`) next to `Prev/Play/Next`; pressing Enter jumps to the requested frame.
- Added manual mask range copy with explicit `Prev`/`Next` frame counts in the `Manual Masks` rollout.
- Fixed append-inference folder/video runs so subsequent prompt-only append passes continue frame-by-frame and merge with existing masks without stalling on frame one.
- Optimized append merging to avoid re-caching all prior frame masks on each append run.
- Fixed compact cache overwrite invalidation so edited manual masks copied across ranges immediately reflect the newest mask content on target frames.
- Finalized frame-scoped filter persistence and export behavior: mask export now always honors per-frame class/ID/instance filters in `View/Export`.

### 2026-03-21

- Added compact v2 cache archives with cropped, bit-packed mask storage to cut per-frame cache size dramatically.
- Added downscaled inference for image, folder, and video workflows with original-size export restoration.
- Added `inference_image_size` to normalized results and `--inference-scale` to the CLI.
- Updated rendering and export so cached inference masks can stay compact while overlays and exported masks match the original source dimensions.

### 2026-03-20

- Optimized cached mask storage from float32 raster arrays to binary masks and documented official SAM 3 checkpoint setup.
- Added project credits to the README and repository metadata.
- Published the runtime-focused `ultralyticsSAM3_GUI` repo structure with GUI, backend, CLI, and export pipeline.
- Added disk-backed cache selection, reusable backend/model loading, and manual mask tooling in the GUI.
- Added image, folder, and video workflows with text, point, box, and mask prompting plus export controls.
