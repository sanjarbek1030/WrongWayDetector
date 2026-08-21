"""
Wrong-Way Driver Detector
==========================
Detects vehicles moving against the expected direction of traffic flow using
YOLOv8 object detection combined with ByteTrack multi-object tracking.

Traffic convention assumed in this script:
    - Legal flow direction is UP the screen (decreasing Y pixel coordinate),
      e.g. vehicles entering from the bottom of the frame and exiting at the top.
    - A vehicle whose Y coordinate consistently INCREASES (moving DOWN the
      screen) for more than WRONG_WAY_FRAME_THRESHOLD consecutive frames is
      flagged as driving the wrong way.

Requirements:
    pip install ultralytics opencv-python numpy

Usage:
    Set VIDEO_SOURCE below to a video file path or an integer (0) for webcam,
    then run this file directly in PyCharm (Run 'wrong_way_detector').

Author: Computer Vision Engineering Team
"""

import time
from collections import defaultdict, deque

import cv2
import numpy as np
from ultralytics import YOLO

# ----------------------------------------------------------------------------
# CONFIGURATION
# ----------------------------------------------------------------------------

# Path to a video file, or an int (e.g. 0) to use a connected webcam.
VIDEO_SOURCE = "traffic_video.mp4"

# Pretrained YOLOv8 nano model -- chosen for real-time inference speed.
MODEL_WEIGHTS = "yolov8n.pt"

# COCO class IDs for vehicles we care about:
# 2 = car, 3 = motorcycle, 5 = bus, 7 = truck
TARGET_CLASS_IDS = {2, 3, 5, 7}

# Minimum detection confidence to keep a box.
CONFIDENCE_THRESHOLD = 0.35

# How many past centroid positions to remember per track ID.
# Also caps memory usage -- old points beyond this length are auto-discarded
# by the deque's maxlen, preventing unbounded growth over a long video.
TRACK_HISTORY_LENGTH = 30

# Number of consecutive "moving down" frames required before a vehicle is
# officially flagged as a wrong-way driver.
WRONG_WAY_FRAME_THRESHOLD = 10

# Minimum pixel displacement between two consecutive frames for that frame's
# motion to be counted as a meaningful (non-jitter) directional step.
MIN_DISPLACEMENT_PX = 1.5

# Number of frames a track can go "unseen" before its history is purged.
# Prevents the tracking_history / violation dictionaries from leaking memory
# when vehicles leave the frame and their track IDs are never reused.
STALE_TRACK_TIMEOUT_FRAMES = 60

# Visual settings
COLOR_LEGAL = (0, 200, 0)        # green (BGR)
COLOR_WRONG_WAY = (0, 0, 255)    # red (BGR)
COLOR_TRAIL = (0, 0, 255)        # red trail for violators
BOX_THICKNESS_LEGAL = 2
BOX_THICKNESS_WRONG_WAY = 4
FLASH_INTERVAL_SEC = 0.4         # how fast the "WRONG WAY!" text blinks

# ----------------------------------------------------------------------------
# STATE CONTAINERS
# ----------------------------------------------------------------------------

# track_id -> deque of (x_centroid, y_centroid) tuples, most recent last.
tracking_history = defaultdict(lambda: deque(maxlen=TRACK_HISTORY_LENGTH))

# track_id -> int, running count of consecutive frames where the vehicle's
# centroid Y coordinate increased (i.e. moved DOWN the screen).
wrong_way_streak = defaultdict(int)

# track_id -> bool, sticky flag once a vehicle is confirmed wrong-way. Sticky
# so the alert doesn't flicker off if the vehicle has one noisy frame.
confirmed_wrong_way = defaultdict(bool)

# track_id -> int, the last frame index this ID was observed in. Used to
# evict stale entries from all the dictionaries above.
last_seen_frame = defaultdict(int)


def cleanup_stale_tracks(current_frame_index):
    """
    Remove bookkeeping data for track IDs that have not been seen recently.

    Without this step, tracking_history / wrong_way_streak / confirmed_wrong_way
    would grow indefinitely over a long-running video feed as new vehicles
    continuously enter and old ones leave -- a classic memory leak in
    long-lived tracking applications.
    """
    stale_ids = [
        track_id
        for track_id, last_frame in last_seen_frame.items()
        if current_frame_index - last_frame > STALE_TRACK_TIMEOUT_FRAMES
    ]
    for track_id in stale_ids:
        tracking_history.pop(track_id, None)
        wrong_way_streak.pop(track_id, None)
        confirmed_wrong_way.pop(track_id, None)
        last_seen_frame.pop(track_id, None)


def update_direction_state(track_id, centroid_y):
    """
    Update the wrong-way streak counter for a single track based on its
    latest Y centroid, and return whether it is currently confirmed
    wrong-way.

    Coordinate math reminder:
        In image space, Y increases DOWNWARD from the top-left origin.
        - "Legal" flow is UP the screen  -> Y should DECREASE over time.
        - "Wrong way" is DOWN the screen -> Y INCREASES over time.
    """
    history = tracking_history[track_id]

    if len(history) < 2:
        # Not enough history yet to determine a direction -- nothing to do.
        return confirmed_wrong_way[track_id]

    previous_y = history[-2][1]
    current_y = history[-1][1]
    delta_y = current_y - previous_y  # positive delta_y == moving DOWN

    if delta_y > MIN_DISPLACEMENT_PX:
        # Vehicle moved down this frame -> extend the wrong-way streak.
        wrong_way_streak[track_id] += 1
    elif delta_y < -MIN_DISPLACEMENT_PX:
        # Vehicle moved up (legal direction) -> reset the streak.
        wrong_way_streak[track_id] = 0
    # else: displacement too small to be meaningful (jitter/stationary),
    # streak is left unchanged so brief noise doesn't reset a real violation.

    if wrong_way_streak[track_id] > WRONG_WAY_FRAME_THRESHOLD:
        confirmed_wrong_way[track_id] = True

    return confirmed_wrong_way[track_id]


def draw_flashing_alert(frame, text, x, y):
    """
    Draw a flashing 'WRONG WAY!' label above a bounding box. The flash is
    driven by wall-clock time so its blink rate is independent of the
    video's frame rate.
    """
    # Toggle visibility based on the current time, producing a blink effect.
    is_visible = int(time.time() / FLASH_INTERVAL_SEC) % 2 == 0
    if not is_visible:
        return

    font = cv2.FONT_HERSHEY_SIMPLEX
    font_scale = 0.8
    thickness = 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, font_scale, thickness)

    text_x = int(x)
    text_y = max(int(y) - 15, text_h + 5)  # keep label on-screen near top edge

    # Solid red background rectangle so text stays legible over any footage.
    cv2.rectangle(
        frame,
        (text_x - 4, text_y - text_h - 6),
        (text_x + text_w + 4, text_y + 4),
        COLOR_WRONG_WAY,
        cv2.FILLED,
    )
    cv2.putText(
        frame, text, (text_x, text_y), font, font_scale,
        (255, 255, 255), thickness, cv2.LINE_AA,
    )


def draw_motion_trail(frame, history, color, thickness=2):
    """Draw a polyline connecting a track's recent centroid history."""
    points = np.array(history, dtype=np.int32)
    if len(points) < 2:
        return
    cv2.polylines(frame, [points.reshape(-1, 1, 2)], isClosed=False,
                  color=color, thickness=thickness, lineType=cv2.LINE_AA)


def safe_extract_boxes(results):
    """
    Defensively extract boxes, track IDs, and class IDs from a YOLO tracking
    result. Returns empty arrays if tracking failed to assign IDs on this
    frame (e.g. very first frames, or a frame with zero detections) instead
    of raising an exception.
    """
    if results is None or len(results) == 0:
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0,))

    result = results[0]
    boxes_obj = result.boxes

    if boxes_obj is None or boxes_obj.id is None:
        # No tracked detections on this frame (nothing detected, or the
        # tracker hasn't assigned persistent IDs yet).
        return np.empty((0, 4)), np.empty((0,), dtype=int), np.empty((0,), dtype=int), np.empty((0,))

    xyxy = boxes_obj.xyxy.cpu().numpy()
    track_ids = boxes_obj.id.cpu().numpy().astype(int)
    class_ids = boxes_obj.cls.cpu().numpy().astype(int)
    confidences = boxes_obj.conf.cpu().numpy()

    return xyxy, track_ids, class_ids, confidences


def main():
    model = YOLO(MODEL_WEIGHTS)

    cap = cv2.VideoCapture(VIDEO_SOURCE)
    if not cap.isOpened():
        print(f"[ERROR] Could not open video source: {VIDEO_SOURCE}")
        return

    window_name = "Wrong-Way Driver Detector"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    frame_index = 0

    try:
        while True:
            ret, frame = cap.read()
            if not ret or frame is None:
                # Defensive handling of end-of-video / dropped frame / read failure.
                print("[INFO] Video stream ended or frame could not be read.")
                break

            frame_index += 1

            # Run detection + tracking in a single call. persist=True keeps
            # track IDs consistent across frames within this video session.
            try:
                results = model.track(
                    source=frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=list(TARGET_CLASS_IDS),
                    conf=CONFIDENCE_THRESHOLD,
                    verbose=False,
                )
            except Exception as tracking_error:
                # If tracking throws (e.g. transient backend issue), skip this
                # frame gracefully instead of crashing the whole application.
                print(f"[WARNING] Tracking failed on frame {frame_index}: {tracking_error}")
                cv2.imshow(window_name, frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
                continue

            xyxy_boxes, track_ids, class_ids, confidences = safe_extract_boxes(results)

            for box, track_id, class_id, conf in zip(xyxy_boxes, track_ids, class_ids, confidences):
                if class_id not in TARGET_CLASS_IDS:
                    continue  # extra safety filter beyond the `classes=` arg above

                x1, y1, x2, y2 = box
                centroid_x = (x1 + x2) / 2.0
                centroid_y = (y1 + y2) / 2.0

                # Record this position and mark the track as freshly seen.
                tracking_history[track_id].append((centroid_x, centroid_y))
                last_seen_frame[track_id] = frame_index

                is_wrong_way = update_direction_state(track_id, centroid_y)

                x1i, y1i, x2i, y2i = int(x1), int(y1), int(x2), int(y2)

                if is_wrong_way:
                    # --- Violation visuals ---
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), COLOR_WRONG_WAY, BOX_THICKNESS_WRONG_WAY)
                    draw_motion_trail(frame, tracking_history[track_id], COLOR_TRAIL, thickness=2)
                    draw_flashing_alert(frame, "WRONG WAY!", x1i, y1i)
                    label = f"ID {track_id} WRONG-WAY"
                    cv2.putText(frame, label, (x1i, y2i + 20), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, COLOR_WRONG_WAY, 2, cv2.LINE_AA)
                else:
                    # --- Legal vehicle visuals ---
                    cv2.rectangle(frame, (x1i, y1i), (x2i, y2i), COLOR_LEGAL, BOX_THICKNESS_LEGAL)
                    label = f"ID {track_id}"
                    cv2.putText(frame, label, (x1i, max(y1i - 8, 12)), cv2.FONT_HERSHEY_SIMPLEX,
                                0.6, COLOR_LEGAL, 2, cv2.LINE_AA)

            # Periodic memory cleanup for tracks that have disappeared.
            if frame_index % STALE_TRACK_TIMEOUT_FRAMES == 0:
                cleanup_stale_tracks(frame_index)

            # HUD: frame counter + active track count, useful for debugging in PyCharm.
            hud_text = f"Frame: {frame_index} | Active tracks: {len(tracking_history)}"
            cv2.putText(frame, hud_text, (10, 25), cv2.FONT_HERSHEY_SIMPLEX,
                        0.6, (255, 255, 255), 2, cv2.LINE_AA)

            cv2.imshow(window_name, frame)

            # Press 'q' to exit cleanly at any time.
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("[INFO] Quit key pressed. Exiting.")
                break

    finally:
        # Always release resources, even if an unexpected exception occurred.
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
