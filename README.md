# Wrong-Way Driver Detector

A real-time computer vision system that detects vehicles driving against the flow of traffic using **YOLOv8** and **ByteTrack** multi-object tracking.

Built with `ultralytics` + `OpenCV`, this script analyzes video streams frame-by-frame, tracks individual vehicles with persistent IDs, and flags any vehicle that consistently moves in the wrong direction — visually alerting with a flashing warning, a red bounding box, and a motion trail.

---

## Features

- **Real-time vehicle detection** using YOLOv8 Nano (`yolov8n.pt`) for fast inference
- **Persistent multi-object tracking** via ByteTrack (`model.track(..., persist=True)`)
- **Class filtering** — tracks only cars, trucks, buses, and motorcycles
- **Directional violation logic** — flags a vehicle as wrong-way if it moves against the expected flow for more than 10 consecutive frames
- **Rich visual annotations**:
  - ✅ Green box + Track ID for legal vehicles
  - 🚨 Thick red box, flashing **"WRONG WAY!"** alert, and a red motion trail for violators
- **Memory-safe design** — stale track data is automatically purged to prevent memory leaks during long video sessions
- **Defensive error handling** — gracefully handles missing track IDs, empty frames, and end-of-video conditions

---

## How It Works

1. Each frame is passed to `model.track()`, which returns bounding boxes with **persistent track IDs**.
2. The centroid `(x, y)` of each vehicle's bounding box is stored in a rolling history buffer (`deque`).
3. Frame-to-frame movement is calculated:
   - **Y decreasing** → vehicle moving up the screen → ✅ legal direction
   - **Y increasing** → vehicle moving down the screen → ⚠️ contributes to a wrong-way streak
4. If a vehicle's downward streak exceeds **10 frames**, it's flagged and stays flagged (sticky alert) with visual warnings drawn directly on the frame.
5. Track data for vehicles that leave the frame is automatically cleaned up after a timeout, keeping memory usage flat over time.

---

## Requirements

```bash
pip install ultralytics opencv-python numpy
```

Python 3.8+ recommended. On first run, `ultralytics` will automatically download `yolov8n.pt` and the `bytetrack.yaml` tracker config.

---

## Usage

1. Clone or download this repository.
2. Open `wrong_way_detector.py` in PyCharm (or any IDE).
3. Set your video source at the top of the file:

```python
VIDEO_SOURCE = "traffic_video.mp4"   # or 0 for a live webcam
```

4. Run the script. A window will open showing live detection and tracking.
5. Press **`q`** at any time to exit.

> **Note:** This script assumes legal traffic flow moves **up** the screen (decreasing Y). If your camera setup is different, flip the direction logic in `update_direction_state()` accordingly.

---

## Configuration

Key parameters can be tuned at the top of the script:

| Parameter | Description | Default |
|---|---|---|
| `MODEL_WEIGHTS` | YOLOv8 model variant | `yolov8n.pt` |
| `TARGET_CLASS_IDS` | COCO class IDs to track | car, motorcycle, bus, truck |
| `CONFIDENCE_THRESHOLD` | Minimum detection confidence | `0.35` |
| `WRONG_WAY_FRAME_THRESHOLD` | Consecutive frames before flagging | `10` |
| `TRACK_HISTORY_LENGTH` | Centroid history length per track | `30` |
| `STALE_TRACK_TIMEOUT_FRAMES` | Frames before purging inactive tracks | `60` |

---

## Tech Stack

- [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics)
- [ByteTrack](https://github.com/ifzhang/ByteTrack) (via Ultralytics tracker integration)
- OpenCV
- NumPy

---

## Future Improvements

- [ ] Support for angled / non-vertical traffic flow (custom direction vectors per lane)
- [ ] Automatic violation logging with timestamps and snapshots
- [ ] Multi-camera / multi-lane support
- [ ] Alert integration (email, webhook, or siren trigger) on confirmed violations

---

## License

This project is provided as-is for educational and research purposes.
