# Semi-Automatic Offside Detector
**Korea University — Deep Learning Final Project**

A semi-automatic soccer offside detection system that combines three foundation models:
- **YOLOv8s-pose** — Player detection + full-body pose estimation (17 keypoints)
- **BLIP VQA** — Jersey color analysis via visual question answering
- **Qwen3-VL-8B** — Natural language offside verdict explanation

---

## Demo

![Demo Flow](assets/demo.gif)

**Usage (3 steps):**
1. Upload a soccer image
2. Click one **attacking team** player (orange A marker) and one **defending team** player (blue B marker)
3. Select goal direction → press **판독 시작**

The system automatically:
- Detects all players with pose estimation
- Clusters jersey colors (BLIP + k-means, k=2) to assign teams
- Draws the offside line at the most forward defender's position
- Judges each attacking player using their most advanced body part (shoulders, hips, knees, ankles — excluding arms, per FIFA rules)
- Generates a natural language explanation with Qwen3-VL-8B

---

## Installation

### Requirements
- Python 3.10+
- CUDA-capable GPU (recommended: 16GB+ VRAM for Qwen3-VL-8B)

### Setup

```bash
git clone https://github.com/<your-username>/offside-detector.git
cd offside-detector

# Create conda environment (recommended)
conda create -n offside python=3.10
conda activate offside

# Install dependencies
pip install -r requirements.txt
```

### Model Downloads
Models are downloaded automatically from HuggingFace on first run:
| Model | Size | Source |
|---|---|---|
| YOLOv8s-pose | ~22 MB | Ultralytics (auto) |
| BLIP VQA | ~990 MB | `Salesforce/blip-vqa-base` |
| Qwen3-VL-8B | ~16 GB | `Qwen/Qwen3-VL-8B-Instruct` |

---

## Run

```bash
python demo.py
```

Open `http://localhost:7861` in your browser.

---

## Code Structure

```
offside-detector/
├── demo.py                  # Main Gradio web app (entry point)
├── requirements.txt
├── models/
│   ├── pose_detector.py     # YOLOv8s-pose wrapper
│   ├── team_classifier.py   # BLIP VQA + k-means team assignment
│   └── reasoner.py          # Qwen3-VL-8B verdict explanation
└── utils/
    ├── line_tools.py        # Offside line geometry
    └── visualizer.py        # Bounding box / skeleton / crop rendering
```

### `demo.py`
Gradio web UI and main pipeline orchestrator. Handles:
- Image upload and interactive player marking
- Calling each model in sequence
- Rendering annotated results and player crop gallery

### `models/pose_detector.py`
Wraps `ultralytics.YOLO` with `yolov8s-pose.pt`. Returns per-player bounding boxes and 17 COCO keypoints (x, y, confidence). Confidence threshold: 0.2.

### `models/team_classifier.py`
Two-stage team assignment:
1. **BLIP VQA** (`Salesforce/blip-vqa-base`) asks *"What color is the jersey?"* for each player crop → returns color label for display
2. **HSV histogram + k-means (k=2)** clusters all players into exactly 2 teams. The cluster whose centroid is closest to the *pixel color at the user's click location* is assigned as the attack (A) or defend (B) team — no nearest-detection lookup required, making it robust to marker placement.

### `models/reasoner.py`
Sends the annotated result image to `Qwen/Qwen3-VL-8B-Instruct` with a prompt describing the offside setup. Returns a 2–3 sentence natural language explanation of why players are or are not offside.

### `utils/line_tools.py`
- `is_on_goal_side(p1, p2, point, goal_side)`: Direction-independent offside judgment using y-interpolated x-comparison (not cross-product, which varies with click order).
- `extend_line_to_edges()`: Extends a two-point line to image boundaries for rendering.

### `utils/visualizer.py`
- `draw_detections()`: Draws bounding boxes, skeleton (12 COCO connections), and team/verdict labels.
- `make_all_crops()`: Generates zoomed crops for each attacking player with the offside line projected into crop space.

---

## Pipeline Overview

```
Image
  │
  ▼
[YOLOv8s-pose] ──────────────────────► Player bboxes + 17 keypoints
  │
  ▼
[BLIP VQA + k-means] ────────────────► Team A (attack) / Team B (defend)
  │
  ▼
Auto offside line ───────────────────► x = most-forward defender's position
  │
  ▼
FIFA body-part rule ─────────────────► Most advanced non-arm keypoint per attacker
  │
  ▼
Geometric judgment ──────────────────► OFFSIDE / ONSIDE per player
  │
  ▼
[Qwen3-VL-8B] ──────────────────────► Natural language explanation
```
