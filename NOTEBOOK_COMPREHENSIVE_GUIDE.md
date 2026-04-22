# Comprehensive Guide: YOLOv12s Hand Pose Training + Video Tracking

This document explains the complete logic inside the two notebooks in this workspace:

- `cse445-yolov12s-hand-pose-training.ipynb`
- `cse445-yolov12s-video-tracking.ipynb`

It covers:

- Code structure and what each block does
- Theory behind detection, pose estimation, and multi-object tracking
- Hyperparameter choices and why they matter
- Extracted experiment outputs (metrics, runtime, tracker behavior)
- Behind-the-scenes design decisions and practical caveats

---

## 1) Notebook Inventory and Purpose

## 1.1 Training Notebook
File: `cse445-yolov12s-hand-pose-training.ipynb`

Primary objective:

- Train a hand pose model (YOLOv12s-Pose) on a 21-keypoint dataset
- Validate performance with explicit metrics
- Produce visual diagnostics (curves and inference examples)
- Export trained weights for reuse

## 1.2 Tracking Notebook
File: `cse445-yolov12s-video-tracking.ipynb`

Primary objective:

- Load trained weights and run video pose tracking
- Compare ByteTrack and BoT-SORT on the same input video
- Quantify tracking behavior (speed, ID continuity, zero-detection rate)
- Visualize side-by-side outputs and summary charts

---

## 2) End-to-End Pipeline (Big Picture)

1. Setup environment and dependencies (`ultralytics`, `lapx`)
2. Inspect dataset and validate YOLO-pose format
3. Build train/val split and write custom `data.yaml`
4. Visualize keypoints/skeleton overlays for label sanity check
5. Train YOLOv12s-Pose model from scratch
6. Evaluate metrics and plot training curves
7. Run inference on held-out validation images
8. Export best checkpoint
9. Run tracking over full video with ByteTrack and BoT-SORT
10. Compare trackers quantitatively and qualitatively

---

## 3) Training Notebook: Code Walkthrough

## 3.1 Environment Setup

Key actions:

- Installs `ultralytics` and `lapx`
- Calls `ultralytics.checks()` for hardware/software status
- Detects CUDA and chooses device

Why this matters:

- Ensures the runtime can run high-throughput GPU training
- Avoids silent CPU fallback that can dramatically increase training time

Behind the scenes:

- `lapx` is required for tracker association internals (even though this notebook is focused on training, dependency consistency is maintained)

## 3.2 Dataset Inspection and Validation

Key actions:

- Locates dataset in Kaggle input paths
- Reads original `data.yaml`
- Verifies keypoint metadata:
  - `kpt_shape: [21, 3]`
  - one class (`hand`)

Why this matters:

- Pose models require strict keypoint schema consistency
- Incorrect keypoint dimensionality or label formatting causes training instability or failure

Behind the scenes:

- Kaggle input is read-only; write operations must target `/kaggle/working`

## 3.3 Dataset Preparation and YAML Generation

Key actions:

- Creates a 90/10 train/val split
- Copies images + labels into a prepared writable dataset root
- Defines canonical keypoint names (21 landmarks)
- Writes custom YAML with updated train/val paths

Why this matters:

- Stable train/val structure is required for reproducible runs
- Explicit YAML avoids hidden path bugs across environments

Behind the scenes:

- Seeded split (`RANDOM_SEED = 42`) helps reproducibility
- Label/image parity checks reduce missing-file edge cases

## 3.4 Visualization Before Training

Key actions:

- Draws bounding boxes, keypoints, and skeleton connections
- Samples several images from dataset splits

Why this matters:

- Label sanity checks before training can prevent wasting hours on corrupted labels
- Visual confirmation catches flipped indexing and malformed coordinates

## 3.5 Model Construction and Training

Core training call includes:

- `model = YOLO("yolo12s-pose.yaml")`
- `model.train(...)` with tuned arguments

### Training Hyperparameters (Extracted)

| Parameter | Value | Why it matters |
|---|---:|---|
| `epochs` | 50 | Controls optimization horizon |
| `imgsz` | 640 | Balances detail and speed |
| `batch` | 16 | Fits T4 memory while keeping stable gradient estimates |
| `device` | auto CUDA | Uses GPU acceleration |
| `workers` | 4 | Data loader throughput |
| `patience` | 15 | Early stopping on no improvement |
| `save_period` | 10 | Intermediate checkpoint persistence |
| `mosaic` | 1.0 | Strong spatial augmentation |
| `fliplr` | 0.5 | Horizontal variation for robustness |
| `flipud` | 0.0 | Disables vertical flips for realistic hand orientation |
| `pose` | 12.0 | Raises keypoint loss contribution |
| `kobj` | 1.0 | Keypoint objectness weighting |

### Training Outputs (Extracted)

- Training time: **55.1 minutes**
- Epochs trained: **50**

## 3.6 Post-Training Validation (`model.val()`)

The notebook runs explicit validation on the prepared validation split.

### Final Validation Metrics (Extracted)

| Metric | Value |
|---|---:|
| Box mAP@0.50 | 0.9634 |
| Box mAP@0.50:0.95 | 0.7039 |
| Pose mAP@0.50 | 0.2113 |
| Pose mAP@0.50:0.95 | 0.0375 |
| Precision | 0.9585 |
| Recall | 0.9238 |

Interpretation:

- Detection is strong (high box mAP, precision, recall)
- Keypoint localization is significantly harder than box localization in this setup

## 3.7 Curve Plotting and Qualitative Inference

Key actions:

- Loads `results.csv`
- Plots loss and metric curves over epochs
- Runs inference on held-out validation images

Why this matters:

- Curves reveal optimization dynamics (underfit/overfit trends)
- Qualitative visuals reveal keypoint stability that scalar metrics may miss

## 3.8 Weight Export

Key actions:

- Copies `best.pt` (or fallback `last.pt`) to output root

Why this matters:

- Standardized export path simplifies downstream tracking notebook integration

---

## 4) Tracking Notebook: Code Walkthrough

## 4.1 Setup and Input Loading

Key actions:

- Installs required packages
- Loads model checkpoint (`best.pt`)
- Validates source video path and metadata

Extracted video info:

- Frames processed: **1423**
- Source FPS: **25.0**

## 4.2 ByteTrack Configuration and Execution

The notebook writes a YAML file for ByteTrack and runs:

- `model.track(..., tracker=bytetrack_yaml, stream=True)`

### ByteTrack Hyperparameters (Extracted)

| Parameter | Value | Effect |
|---|---:|---|
| `track_high_thresh` | 0.5 | Stage-1 high-confidence association |
| `track_low_thresh` | 0.1 | Low-confidence recovery association |
| `new_track_thresh` | 0.6 | Gate for creating new IDs |
| `track_buffer` | 30 | Track persistence duration |
| `match_thresh` | 0.8 | IoU matching strictness |
| `conf` | 0.30 | Detector confidence threshold |
| `iou` | 0.50 | NMS suppression threshold |
| `imgsz` | 640 | Detection resolution |
| `persist` | True | Keep IDs across frames |
| `stream` | True | Generator mode for memory efficiency |

### Runtime (Extracted)

- ByteTrack complete in **53.7s**
- Throughput: **26.5 FPS**

## 4.3 BoT-SORT Configuration and Execution

BoT-SORT extends baseline tracking with motion compensation and appearance logic options.

### BoT-SORT-Specific Additions (Extracted)

| Parameter | Value | Effect |
|---|---|---|
| `gmc_method` | `sparseOptFlow` | Camera motion compensation |
| `proximity_thresh` | 0.5 | Spatial gating |
| `appearance_thresh` | 0.25 | Appearance match gate |
| `with_reid` | False | Re-ID disabled in this run |

### Runtime (Extracted)

- BoT-SORT complete in **84.3s**
- Throughput: **16.9 FPS**
- Extra runtime vs ByteTrack: **+30.5s**

## 4.4 Side-by-Side Frame Comparison

The notebook extracts synchronized frames from both output videos and builds a side-by-side figure.

Why this matters:

- Visual comparison catches behavior not fully reflected by aggregate metrics
- Helps identify ID persistence and jitter during overlap/occlusion

## 4.5 Quantitative Tracking Analysis Function

A custom function computes:

- Unique IDs issued
- ID switch events
- Average detections per frame
- Track lifetime
- Zero-detection frame count

### Tracker Statistics (Extracted)

| Statistic | ByteTrack | BoT-SORT |
|---|---:|---:|
| Total frames | 1423 | 1423 |
| Unique track IDs | 2 | 2 |
| ID switch events | 18 | 18 |
| Avg detections/frame | 0.88 | 0.88 |
| Max detections/frame | 1 | 1 |
| Avg track lifetime (s) | 25.2 | 25.2 |
| Zero-detection frames | 164 (11.5%) | 164 (11.5%) |

Interpretation:

- In this specific video and parameter setup, tracker continuity metrics are nearly identical
- Performance difference is mainly runtime throughput, favoring ByteTrack

---

## 5) Theory Section

## 5.1 Pose Estimation with YOLO

The pose model predicts both:

- Bounding box for hand localization
- Keypoint coordinates for 21 landmarks

A simplified objective combines detection and keypoint terms:

$$
\mathcal{L}_{total} = \lambda_{box}\mathcal{L}_{box} + \lambda_{cls}\mathcal{L}_{cls} + \lambda_{pose}\mathcal{L}_{pose} + \lambda_{kobj}\mathcal{L}_{kobj}
$$

Where larger `pose` weight increases emphasis on keypoint quality.

## 5.2 mAP, Precision, Recall

- Precision: fraction of predicted positives that are correct
- Recall: fraction of true positives recovered
- mAP@0.50: AP measured at IoU threshold 0.50
- mAP@0.50:0.95: stricter average across multiple IoU thresholds

Higher box mAP and precision/recall with lower pose mAP often means:

- Hand location is correct
- Landmark positions are still noisy or difficult under blur/occlusion

## 5.3 ByteTrack vs BoT-SORT

### ByteTrack

- Associates high-confidence detections first
- Uses low-confidence detections in a second stage to recover tracks
- Typically fast and robust for real-time constraints

### BoT-SORT

- Adds Kalman-based motion prediction
- Adds camera motion compensation
- Can improve continuity in moving-camera scenes but increases compute

---

## 6) Behind-the-Scenes Engineering Decisions

1. **Kaggle path handling**
- Read from `/kaggle/input`
- Write all generated artifacts to `/kaggle/working`

2. **Memory-aware streaming in tracking**
- `stream=True` avoids holding all intermediate tensors at once

3. **State reset between tracker runs**
- Reinitializing model before BoT-SORT avoids cross-run state contamination

4. **Fallback logic for checkpoints**
- Uses `best.pt` when available, otherwise `last.pt`

5. **Plot persistence**
- Saves figures to file paths before/alongside display calls

---

## 7) Practical Limitations and Risks

1. **Pose metrics are low despite strong detection metrics**
- Indicates landmark-level errors under difficult frames

2. **ID-switch counting heuristic is simplified**
- Current heuristic uses new-ID appearances as potential switches; refined MOT metrics could improve rigor

3. **Single-video evaluation scope**
- Tracker conclusions may shift on more crowded or camera-moving videos

4. **No Re-ID in BoT-SORT run**
- `with_reid=False` reduces ability to recover long-term identity consistency after full occlusion

---

## 8) Suggested Improvement Roadmap

1. Improve pose quality
- Add stronger keypoint-focused augmentations
- Increase data diversity for hand articulation and occlusion
- Tune `pose`/`kobj` loss weights systematically

2. Strengthen evaluation
- Add per-keypoint error plots and confidence distributions
- Track trajectory smoothness metrics and MOT-style IDF1/HOTA proxies

3. Extend tracking benchmark
- Evaluate on multiple videos with varying hand counts and camera motion
- Compare `with_reid=True` variant in BoT-SORT where feasible

4. Runtime optimization
- Test lower `imgsz` or frame stride (`vid_stride`) for deployment constraints
- Export model to optimized runtime backend if needed

---

## 9) Final Summary

- The training notebook is well-structured and reproducible, with strong detection metrics and clear diagnostics.
- The tracking notebook provides a fair side-by-side tracker comparison with clear quantitative outputs.
- On this run, ByteTrack offers much better speed while both trackers report similar continuity statistics.
- The largest performance gap is in pose keypoint accuracy, suggesting the next optimization focus should be detector/keypoint quality rather than tracker-only tuning.
