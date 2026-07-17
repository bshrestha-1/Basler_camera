# AI Analysis: YOLO Detection and SAM2 Segmentation

Phase 19 adds AI-based particle detection, classification, and
segmentation on top of the classical blob-detection pipeline from
[`analysis.md`](analysis.md): a trained [YOLO](https://docs.ultralytics.com/)
model detects and classifies every particle (including automatic intruder
identification) even under poor lighting or heavy overlap, and
[SAM2](https://github.com/facebookresearch/sam2) refines each detection
into an exact pixel mask for shape, contact-area, and packing
measurements the classical circular-particle approximation can't provide.

```
Camera frame -> YoloParticleDetector.detect() -> YoloDetection -> ParticleTracker
Camera frame + box prompt -> Sam2Segmenter.segment() -> ParticleSegment -> ShapeMetrics
```

Both models support full training pipelines (dataset preparation,
annotation, configuration, training, validation, checkpoint management,
export) as well as inference-only use with pretrained/downloaded weights
-- a researcher who just wants to run a published model doesn't need to
touch any of the training code.

## Optional dependency

`torch`, `ultralytics`, and `sam2` are a separate, optional dependency
group:

```bash
pip install "glas[ai]"
```

`import glas`, the CLI, and the GUI all work with none of these
installed -- every AI module lazy-imports them, and any AI feature used
without the extra installed raises `glas.exceptions.AIDependencyError`
(CLI) or shows a dialog naming exactly which packages are missing and the
install command that fixes it (GUI), rather than failing with a raw
`ImportError` at import time. `glas.ai.dependencies.missing_ai_packages()`
reports which of the three are absent; `describe_missing_ai_packages()`
turns that into the human-readable message both surfaces use.

## Module layout

- `glas.ai.dependencies` -- lazy-import helpers for `torch`,
  `ultralytics`, and `sam2`, and the missing-dependency detection/message
  functions above. Every other `glas.ai` module and the GUI/CLI AI code
  paths go through this module, never `import torch` directly.
- `glas.ai.yolo_detector` -- `YoloParticleDetector` (wraps a trained
  `ultralytics` model for inference), `YoloDetection` (a
  `~glas.analysis.tracking_utils.Detection` subclass carrying label,
  confidence, and intruder flag), and `track_dataset_yolo()` (the YOLO
  equivalent of `glas.analysis.track_dataset()`).
- `glas.ai.yolo_train` -- `train_yolo()`, `validate_yolo()`,
  `export_yolo_model()`: thin wrappers around `ultralytics`'s own
  training loop, plus `YoloTrainingConfig`/`YoloTrainingResult` for typed
  config and results.
- `glas.ai.annotation` -- `auto_annotate_dataset()` bootstraps YOLO
  training boxes from a recording via the existing classical blob
  detector; `prepare_yolo_dataset()` splits labeled frames into
  train/val and writes a YOLO-format `data.yaml`.
- `glas.ai.sam2_segmenter` -- `Sam2Segmenter` (wraps a SAM2 image
  predictor for box-prompted mask segmentation), `ParticleSegment`
  (mask + score + shape metrics), `compute_shape_metrics()` (area,
  perimeter, centroid, orientation, aspect ratio), `compute_contact_area()`
  (shared boundary between two touching particles), and
  `compute_segmentation_summary()` (packing fraction, void fraction,
  pairwise contacts for a whole frame).
- `glas.ai.sam2_train` -- `train_sam2()`: lightweight fine-tuning (freeze
  the image encoder, train only the prompt encoder and mask decoder) on
  box-prompted ground-truth masks; `auto_annotate_masks()` bootstraps
  those masks via classical contour detection; `prepare_sam2_dataset()`
  writes a training manifest.

## Quickstart: inference only (pretrained/existing weights)

```python
from pathlib import Path
from glas.ai.yolo_detector import track_dataset_yolo

history = track_dataset_yolo(Path("recordings/run_001"), "glass_beads.pt")
for track_id, observations in history.items():
    last = observations[-1]
    print(track_id, last.label, last.confidence, last.is_intruder)
```

```python
from glas.ai.sam2_segmenter import Sam2Segmenter, compute_segmentation_summary
from glas.analysis.tracking_utils import detect_particles
from glas.dataset import iter_frames

frames = list(iter_frames(Path("recordings/run_001")))
last_frame = frames[-1]
detections = detect_particles(last_frame.image)

segmenter = Sam2Segmenter(model_id="facebook/sam2.1-hiera-large")
segments = segmenter.segment_frame(last_frame.image, detections)
summary = compute_segmentation_summary(segments, last_frame.image.shape[:2])
print(summary.packing_fraction, summary.void_fraction, len(summary.contacts))
```

## Quickstart: training a custom YOLO detector

```python
from pathlib import Path
from glas.ai.annotation import auto_annotate_dataset, prepare_yolo_dataset
from glas.ai.yolo_train import YoloTrainingConfig, train_yolo

# 1. Bootstrap boxes from a recording via classical blob detection --
#    hand-correct labels/boxes with an external annotation tool for
#    better-than-bootstrap accuracy.
annotations = auto_annotate_dataset(
    Path("recordings/run_001"), Path("staging"), label="glass_bead"
)
data_yaml = prepare_yolo_dataset(annotations, ["glass_bead"], Path("yolo_dataset"))

# 2. Train.
config = YoloTrainingConfig(data_yaml=data_yaml, epochs=100)
result = train_yolo(config)
print(result.best_weights, result.metrics)
```

## Quickstart: fine-tuning SAM2

```python
from pathlib import Path
from glas.ai.sam2_train import auto_annotate_masks, prepare_sam2_dataset, Sam2TrainingConfig, train_sam2

examples = auto_annotate_masks(Path("recordings/run_001"), Path("staging"))
manifest_path = prepare_sam2_dataset(examples, Path("sam2_dataset"))

config = Sam2TrainingConfig(
    manifest_path=manifest_path,
    base_config_file="configs/sam2.1/sam2.1_hiera_l.yaml",
    base_checkpoint_path=Path("sam2.1_hiera_large.pt"),
    output_checkpoint_path=Path("glas_sam2_finetuned.pt"),
    epochs=20,
)
result = train_sam2(config)
print(result.checkpoint_path, result.metrics)
```

## CLI

```bash
# Detect, classify, and track particles with a trained YOLO model.
glas ai detect recordings/run_001 glass_beads.pt --csv tracks.csv

# Bootstrap and train a YOLO dataset from a recording.
glas ai prepare-yolo-dataset recordings/run_001 yolo_dataset --label glass_bead
glas ai train-yolo yolo_dataset/data.yaml --epochs 100

# Segment every particle's exact outline in a dataset's last frame.
glas ai segment recordings/run_001 glass_beads.pt --model-id facebook/sam2.1-hiera-large --csv shapes.csv

# Bootstrap and fine-tune a SAM2 dataset from a recording.
glas ai prepare-sam2-dataset recordings/run_001 sam2_dataset
glas ai train-sam2 sam2_dataset/manifest.json configs/sam2.1/sam2.1_hiera_l.yaml sam2.1_hiera_large.pt glas_sam2_finetuned.pt
```

Run `glas ai --help` for the full option list of every subcommand. Every
`glas ai` command that hits a missing dependency prints exactly which
packages to install and exits with a nonzero status, rather than crashing
with an `ImportError` traceback.

## GUI

The desktop GUI's analysis panel ([`gui.md`](gui.md)) has two AI-backed
tabs alongside the classical ones: **Detection (YOLO)** (dataset folder +
weights path) and **Segmentation (SAM2)** (dataset folder + SAM2 model
id, defaulting to `facebook/sam2.1-hiera-large` so it works out of the box
with nothing but a recording). Both run on the same background
`QThread` as every other analysis tab. If `torch`/`ultralytics`/`sam2`
aren't installed, clicking Run shows a modal dialog naming the missing
packages and the `pip install "glas[ai]"` fix instead of starting a
background run that would only fail.

## Integration with existing tracking and export

`YoloDetection` is a `~glas.analysis.tracking_utils.Detection` subclass,
so YOLO output plugs directly into the existing
`~glas.analysis.particle_tracking.ParticleTracker` with no changes to
tracking logic at all -- `track_dataset_yolo()` produces the exact same
`dict[int, list[TrackedParticle]]` shape as classical `track_dataset()`,
just with `label`/`confidence`/`is_intruder` populated on every
observation instead of left at their classical defaults
(`None`/`None`/`False`). That means every downstream pipeline that
already consumes tracking history --
[Brazil nut](brazil-nut.md), [packing](packing.md), and
[segregation](segregation.md) -- works unchanged with YOLO-sourced
tracks.

`glas.analysis.export_tracks_csv()` writes the same CSV format
(`track_id, frame_id, x, y, radius, area, host_timestamp_ns, label,
confidence, is_intruder`) for both classical and YOLO tracking output --
`label`/`confidence` are simply blank for classical tracking, so a
downstream script never needs to know which one produced a given file.
