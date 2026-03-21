# sam3_ultralytics

`sam3_ultralytics` is an API-first Python package that wraps Ultralytics SAM 3 predictors behind a stable backend, a compact PySide6 desktop GUI, and a CLI.

The package is designed around a single backend class, `SAM3Ultralytics`, so the GUI, CLI, scripts, and tests all use the same inference and export logic.

## Installation

Create or activate an environment with Python 3.11+ and install the package:

```bash
pip install -e .
```

Ultralytics SAM 3 image and video semantic inference currently also require:

```bash
pip install timm
pip install "git+https://github.com/ultralytics/CLIP.git"
```

## Supported Prompt Types

Supported directly through the current Ultralytics SAM 3 API:

- Text prompts for image segmentation
- Text plus box prompts for mixed image prompting
- Point prompts for interactive image segmentation
- Box prompts for interactive image segmentation
- Mask prompts for interactive image segmentation
- Text or box initialization for video tracking
- Point, box, and mask initialization for interactive video tracking

Compatibility refinement paths in this package:

- Mask plus text for image inference runs a semantic text pass and refines with the interactive predictor.
- Mask plus text for video tracking runs a semantic first-frame text pass and initializes interactive tracking from the resulting geometry.
- Mask plus points and mask plus boxes are forwarded through Ultralytics interactive predictors after internal mask normalization.

Backend compatibility boundary:

- Exemplar image prompting is modeled as an adapter boundary because the current local Ultralytics SAM 3 build exposes text, point, box, and mask flows publicly, but not a stable public exemplar API. The package keeps exemplar support isolated so an official adapter can be dropped in cleanly.

## Python API Usage

```python
from sam3_ultralytics import SAM3Ultralytics

backend = SAM3Ultralytics(r"D:\cache\models\sam3.pt", device="auto")
backend.load()

result = backend.predict_image(
    r"D:\deepDataSystems\02_projects\02_nola\02_scans\01_jacksonSquare\02_Qoocam\test\jacksonSquare_qoocam_00001.jpg",
    text_prompt="person",
    output_dir="outputs",
    export_mask_dir="outputs/masks",
)

print(result.labels)
print(result.scores)
print(result.timings)
```

Mask refinement on an image:

```python
result = backend.predict_image(
    "image.jpg",
    text_prompt="person",
    mask_input="rough_mask.png",
    points=[(500, 375, 1)],
)
```

Directory batch processing:

```python
results = backend.predict_batch(
    [r"D:\deepDataSystems\02_projects\02_nola\02_scans\01_jacksonSquare\02_Qoocam\test"],
    text_prompt="person",
    export_mask_dir="outputs/masks",
)
```

Video tracking with mask initialization:

```python
results = backend.track_video(
    "video.mp4",
    mask_input="frame0_mask.png",
    export_mask_dir="video_outputs/masks",
    annotated_video_path="video_outputs/annotated_video.mp4",
)
```

## CLI Usage

Image prediction:

```bash
sam3-ultralytics image \
  --model "D:\cache\models\sam3.pt" \
  --device cuda:0 \
  --text person \
  --mask rough_mask.png \
  --output-dir outputs \
  --mask-dir outputs\masks \
  "image.jpg"
```

Directory batch prediction:

```bash
sam3-ultralytics image \
  --model "D:\cache\models\sam3.pt" \
  --text person \
  --mask-dir outputs\masks \
  "D:\deepDataSystems\02_projects\02_nola\02_scans\01_jacksonSquare\02_Qoocam\test"
```

Video tracking:

```bash
sam3-ultralytics video-track \
  --model "D:\cache\models\sam3.pt" \
  --device cuda:0 \
  --mask frame0_mask.png \
  --output-dir outputs \
  --mask-dir outputs\masks \
  --annotated-video outputs\annotated_video.mp4 \
  "video.mp4"
```

Batch prediction:

```bash
sam3-ultralytics batch \
  --model "D:\cache\models\sam3.pt" \
  --text person \
  image1.jpg image2.jpg clip.mp4
```

## GUI Launch and Usage

Launch the GUI with:

```bash
python -m sam3_ultralytics.gui
```

Installed entry point:

```bash
sam3-gui
```

GUI workflow:

1. Select a SAM 3 checkpoint and device.
2. Open an image, a folder of images, or a video.
3. Add a text prompt and optionally load an exemplar or mask file.
4. Add point prompts or box prompts when interactive refinement is needed.
5. Optionally choose an export directory and enable auto-export masks.
6. Run inference.
7. Review the preview, result panel, playback controls, and export masks on demand.

## Keyboard Shortcuts

Current GUI shortcuts are intentionally minimal:

- `Esc`: use the window manager close behavior
- Mouse click on preview: add point when Point Tool is active
- Mouse drag on preview: add box when Box Tool is active

## Output Schema

Every normalized result exposes:

- `masks`
- `boxes`
- `scores`
- `labels`
- `track_ids`
- `frame_index`
- `source`
- `prompt_metadata`
- `timings`

Objects are serialized without embedding raw mask arrays in JSON output. Mask PNG paths are included when masks are exported.

Mask prompts are included in `prompt_metadata` for reproducibility, including source, normalized shape, mask count, dtype normalization, and whether the mask was treated as binary or probabilistic.

## Export Directory Behavior

Image exports use deterministic names:

- `<source_stem>_mask_<index>.png`
- `<source_stem>_overlay.png`
- `<source_stem>_results.json`
- `<source_stem>_merged_mask.png`

When track ids are present, mask names prefer the track id:

- `<source_stem>_track_012.png`

Video exports use a predictable structure:

- `<export_dir>/frames/frame_000001_overlay.png`
- `<export_dir>/masks/frame_000001_obj_001.png`
- `<export_dir>/masks/frame_000001_track_012.png`
- `<export_dir>/json/frame_000001.json`
- `<export_dir>/annotated_video.mp4`

Directory batch exports reuse the same deterministic image naming per source image.

Export directories are validated and created when possible. Non-directory or non-writable paths raise a clear export error.

## Image and Video Examples

Sample local assets used during development:

- Model: `D:\cache\models\sam3.pt`
- Images: `D:\deepDataSystems\02_projects\02_nola\02_scans\01_jacksonSquare\02_Qoocam\test`

Text-prompt image inference on the sample image was verified locally through `SAM3SemanticPredictor`.

Mask-prompt image and video compatibility flows are covered by the test suite with mocked Ultralytics predictors.

## Limitations

- The current Ultralytics SAM 3 semantic path still does not expose native text-plus-point prompting.
- Text-plus-mask uses an explicit compatibility workflow that sequences native Ultralytics semantic and interactive predictors instead of relying on undocumented joint prompt behavior.
- The current local Ultralytics SAM 3 build does not expose a stable public exemplar image API. The package keeps exemplar handling behind an adapter interface instead of inventing unsupported model behavior.
- The current Ultralytics interactive SAM 3 path requires `compile=None`. This package normalizes that at load time to avoid an upstream `torch.compile` and Triton failure path.
- Full video results keep per-frame masks in memory. Large videos may require a streaming or chunked persistence strategy in a future revision.

## Troubleshooting

### PySide6 GUI dependencies

If the GUI does not launch, install PySide6 explicitly:

```bash
pip install PySide6
```

### Missing SAM 3 runtime dependencies

If SAM 3 image or video semantic inference fails with missing modules such as `clip` or `timm`, install them in the same environment used to run the package:

```bash
python -m pip install timm
python -m pip install "git+https://github.com/ultralytics/CLIP.git"
```

### GPU usage

Use `device="cuda:0"` or choose `cuda:0` in the GUI device selector. Verify PyTorch sees CUDA in the active environment:

```bash
python -c "import torch; print(torch.cuda.is_available(), torch.cuda.get_device_name(0))"
```

### Ultralytics config permissions on Windows

This package sets `YOLO_CONFIG_DIR` to a writable location to avoid Windows roaming-profile permission errors around `Ultralytics/settings.json`.
