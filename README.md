# ultralyticsSAM3_GUI

`ultralyticsSAM3_GUI` is a desktop-first SAM 3 application built on top of Ultralytics SAM 3. It provides a compact PySide6 GUI for image, folder, and video workflows, plus a reusable Python backend and CLI for automation and scripting.

The project is structured so the GUI, CLI, and backend all use the same `SAM3Ultralytics` runtime layer.

## Credits

Created by **Jeffrey Ian Wilson** with implementation support from **OpenAI Codex**.

Additional project credit details are listed in [CREDITS.md](CREDITS.md).

## Features

- Ultralytics SAM 3 backend wrapper with a stable Python API
- PySide6 desktop GUI optimized for laptop workflows
- Text, point, box, and mask prompting for inference
- Manual mask authoring as a separate non-inference layer
- Image, folder, and video support
- Overlay, JSON, and raw mask export
- Cache-backed mask and result storage
- Reusable loaded model instance across runs for lower overhead

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
- `Clear Manual Mask`

### 5. Export

Inside the `Export` rollout:

- `Export Directory`
- `Auto-export masks after inference`
- `Export merged masks only`
- `Invert exported masks`
- `Mask Dilation (px)`
- `Export Masks`

### 6. View Controls

Inside the `View` rollout:

- overlay opacity
- label visibility
- mask visibility
- track ID visibility
- class filter
- ID filter

## Sequence and Video Behavior

### Folder Runs

Folder runs are processed one item at a time through the GUI:

1. move the view to the current image
2. run inference for that image
3. load the result into the view
4. advance to the next image

Folder runs are independent per-image inference. They do not use hidden tracking across the folder.

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
sam3-ultralytics image --model "D:\cache\models\sam3.pt" --text person --all-items "D:\images"
```

Video tracking:

```powershell
sam3-ultralytics video-track --model "D:\cache\models\sam3.pt" --mask "D:\masks\frame0.png" "D:\clips\clip.mp4"
```

Mixed batch:

```powershell
sam3-ultralytics batch --model "D:\cache\models\sam3.pt" --text person "D:\images" "D:\clips\clip.mp4"
```

## Caching and Memory Behavior

The GUI supports a disk-backed cache directory.

What is cached:

- prompt masks
- manual masks
- inference result masks
- Ultralytics writable settings/config under the cache directory

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
