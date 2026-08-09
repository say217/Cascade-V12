"""
Flood Detection + Segmentation Live Viewer (CPU-smooth, adaptive frame skip)
------------------------------------------------------------------------------
Runs a trained YOLOv8 model on a local video file and shows TWO live windows:

  1) "Detection"    -> thin bounding boxes + confidence label/bar
  2) "Segmentation" -> best-effort pixel-level flood-area highlight

Segmentation source is chosen automatically:
  - YOLOv8-SEGMENTATION checkpoint (model.task == "segment") -> real masks.
  - Plain DETECTION checkpoint (model.task == "detect")      -> HSV
    color-heuristic inside each box (fast approximation, clearly logged).

WHY THIS VERSION DOESN'T LAG ON CPU
------------------------------------
A fixed "process every Nth frame" skip still falls behind over time if
inference is slower than the video's real frame rate -- the backlog just
grows. This version instead:
  1. Times how long inference+draw actually takes (EMA-smoothed).
  2. Computes how many source frames elapse in that time.
  3. Uses cap.grab() (cheap: no decode/color-convert/copy) to fast-forward
     through exactly that many frames before decoding+processing the next
     one with cap.retrieve()/read().
This keeps the live view roughly synced to real playback speed instead of
queuing up a growing backlog -- the classic cause of "lag."

Controls while running:
    q      -> quit
    space  -> pause / resume
"""

import os
import time
from pathlib import Path

import cv2
import numpy as np
import torch
from ultralytics import YOLO

PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODEL_PATH = PROJECT_ROOT / ".model" / "DeltaFLOODM9.pt"
VIDEO_PATH = (
    PROJECT_ROOT
    / "Assets"
    / "test_vedio"
    / "vidssave.com Drone footage of flood damage in Aceh as Indonesia steps up response _ AFP 720P.mp4"
)

OUTPUT_DIR = PROJECT_ROOT / "outputs"
OUTPUT_DET_PATH = OUTPUT_DIR / "flood_detection_output.mp4"
OUTPUT_SEG_PATH = OUTPUT_DIR / "flood_segmentation_output.mp4"

# ----------------------------------------------------------------------
# CPU-friendly inference / display settings
# ----------------------------------------------------------------------
IMG_SIZE       = 384    # lower than 480/640 -> meaningfully faster on CPU
CONF_THRES     = 0.25
MIN_SKIP       = 0      # manual floor: always skip at least this many frames
DISPLAY_SCALE  = 0.6    # shrink preview windows so imshow itself is cheap
SAVE_OUTPUT    = True   # set False to skip writing annotated .mp4 files
EMA_ALPHA      = 0.25   # smoothing for the adaptive-skip timing estimate

BOX_COLOR         = (255, 255, 255)
BADGE_COLOR       = (28, 28, 28)
BAR_BG_COLOR      = (85, 85, 85)
SEG_OVERLAY_COLOR = (255, 140, 0)   # BGR -> renders as orange on the seg view

torch.set_num_threads(os.cpu_count() or 4)  # use all CPU cores for conv ops


def load_model(model_path: Path) -> YOLO:
    if not model_path.exists():
        raise FileNotFoundError(f"Model checkpoint not found: {model_path.resolve()}")
    model = YOLO(str(model_path))
    model.to("cpu")
    print(f"Loaded model | task = '{model.task}' | device = cpu")
    return model


def open_video(video_path: Path) -> cv2.VideoCapture:
    if not video_path.exists():
        raise FileNotFoundError(f"Video file not found: {video_path.resolve()}")
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise IOError(f"OpenCV could not open video: {video_path}")
    return cap


def draw_detections(frame, boxes):
    """Thin white box + label + confidence bar. Mutates and returns `frame`."""
    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        conf = float(box.conf[0])

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thickness=1)

        label_text = f"flood {conf * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(
            label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.4, 1
        )

        text_x = x1 + 2
        text_y = y1 - 6 if y1 - 6 > 15 else y1 + text_h + 4

        cv2.rectangle(
            frame, (x1, text_y - text_h - 2), (x2, text_y + baseline), BADGE_COLOR, -1
        )
        cv2.putText(
            frame, label_text, (text_x, text_y),
            cv2.FONT_HERSHEY_SIMPLEX, 0.4, BOX_COLOR, thickness=1, lineType=cv2.LINE_AA,
        )

        bar_x_start = text_x + text_w + 2
        bar_max_width = max(30, (x2 - bar_x_start) - 4)
        bar_width = int(bar_max_width * conf)
        bar_height = 5
        bar_y = text_y - text_h + 2

        cv2.rectangle(
            frame, (bar_x_start, bar_y),
            (bar_x_start + bar_max_width, bar_y + bar_height), BAR_BG_COLOR, -1,
        )
        cv2.rectangle(
            frame, (bar_x_start, bar_y),
            (bar_x_start + bar_width, bar_y + bar_height), BOX_COLOR, -1,
        )
    return frame


def segmentation_from_masks(frame, result):
    """Uses real predicted masks -- only available for a YOLOv8-seg checkpoint."""
    h, w = frame.shape[:2]
    overlay = np.zeros_like(frame)

    masks = result.masks.data.cpu().numpy()  # (N, mask_h, mask_w)
    for m in masks:
        mask_resized = cv2.resize(m, (w, h), interpolation=cv2.INTER_LINEAR)
        overlay[mask_resized > 0.5] = SEG_OVERLAY_COLOR

    seg_view = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    contour_src = (overlay.sum(axis=2) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_view, contours, -1, BOX_COLOR, 1)
    return seg_view


def segmentation_from_color_heuristic(frame, boxes):
    """
    Fallback for plain DETECTION checkpoints (no masks available).
    Approximates the flood-water region inside each detected box using an
    HSV color band typical of murky/muddy floodwater, cleaned with
    morphology. Fast enough for real-time CPU playback, but it's an
    approximation -- not a learned segmentation.
    """
    overlay = np.zeros_like(frame)
    hsv = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)

    lower = np.array([0, 10, 40])
    upper = np.array([40, 140, 200])
    water_mask_full = cv2.inRange(hsv, lower, upper)

    for box in boxes:
        x1, y1, x2, y2 = map(int, box.xyxy[0])
        x1, y1 = max(x1, 0), max(y1, 0)
        x2, y2 = min(x2, frame.shape[1]), min(y2, frame.shape[0])
        if x2 <= x1 or y2 <= y1:
            continue

        roi_mask = np.zeros(water_mask_full.shape, dtype=np.uint8)
        roi_mask[y1:y2, x1:x2] = water_mask_full[y1:y2, x1:x2]
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_OPEN, np.ones((5, 5), np.uint8))
        roi_mask = cv2.morphologyEx(roi_mask, cv2.MORPH_CLOSE, np.ones((9, 9), np.uint8))

        overlay[roi_mask > 0] = SEG_OVERLAY_COLOR

    seg_view = cv2.addWeighted(overlay, 0.45, frame, 0.55, 0)
    contour_src = (overlay.sum(axis=2) > 0).astype(np.uint8) * 255
    contours, _ = cv2.findContours(contour_src, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    cv2.drawContours(seg_view, contours, -1, BOX_COLOR, 1)
    return seg_view


def main():
    if SAVE_OUTPUT:
        OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    model = load_model(MODEL_PATH)
    is_seg_model = model.task == "segment"
    print(f"Segmentation source: {'model masks' if is_seg_model else 'color heuristic (detector-only checkpoint)'}")

    cap = open_video(VIDEO_PATH)
    src_fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    print(f"Video: {VIDEO_PATH.name} | {width}x{height} @ {src_fps:.1f}fps | {total_frames} frames")

    target_frame_time = 1.0 / max(src_fps, 1e-3)  # seconds "used up" per source frame at real speed

    out_det = out_seg = None
    if SAVE_OUTPUT:
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        out_det = cv2.VideoWriter(str(OUTPUT_DET_PATH), fourcc, src_fps, (width, height))
        out_seg = cv2.VideoWriter(str(OUTPUT_SEG_PATH), fourcc, src_fps, (width, height))

    cv2.namedWindow("Detection", cv2.WINDOW_NORMAL)
    cv2.namedWindow("Segmentation", cv2.WINDOW_NORMAL)

    frame_idx = 0
    paused = False
    t_loop_start = time.time()
    proc_time_ema = None
    skip_count = MIN_SKIP

    while True:
        if not paused:
            # Cheap fast-forward: grab() decodes without the color-convert +
            # copy that read() does, so skipped frames cost much less.
            for _ in range(skip_count):
                if not cap.grab():
                    break
                frame_idx += 1

            ret, frame = cap.read()
            if not ret:
                break
            frame_idx += 1

            t0 = time.time()
            results = model.predict(
                source=frame, imgsz=IMG_SIZE, conf=CONF_THRES, device="cpu", verbose=False
            )
            result = results[0]
            boxes = result.boxes

            det_view = draw_detections(frame.copy(), boxes)
            if is_seg_model and result.masks is not None:
                seg_view = segmentation_from_masks(frame, result)
            else:
                seg_view = segmentation_from_color_heuristic(frame, boxes)
            proc_time = time.time() - t0

            # Adaptive skip: if processing this frame took N source-frame-durations,
            # skip roughly N frames next time so we don't build up a backlog.
            proc_time_ema = proc_time if proc_time_ema is None else (
                EMA_ALPHA * proc_time + (1 - EMA_ALPHA) * proc_time_ema
            )
            skip_count = max(MIN_SKIP, int(proc_time_ema / target_frame_time) - 1)

            if SAVE_OUTPUT:
                out_det.write(det_view)
                out_seg.write(seg_view)

            elapsed = time.time() - t_loop_start
            live_fps = frame_idx / elapsed if elapsed > 0 else 0.0

            det_disp = cv2.resize(det_view, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            seg_disp = cv2.resize(seg_view, None, fx=DISPLAY_SCALE, fy=DISPLAY_SCALE)
            cv2.putText(
                det_disp, f"{live_fps:.1f} fps (cpu) | auto-skip {skip_count}",
                (8, 18), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA,
            )

            cv2.imshow("Detection", det_disp)
            cv2.imshow("Segmentation", seg_disp)

        key = cv2.waitKey(1) & 0xFF
        if key == ord("q"):
            break
        elif key == ord(" "):
            paused = not paused

    cap.release()
    if SAVE_OUTPUT:
        out_det.release()
        out_seg.release()
    cv2.destroyAllWindows()

    print("Done.")
    if SAVE_OUTPUT:
        print(f"Saved: {OUTPUT_DET_PATH.resolve()}")
        print(f"Saved: {OUTPUT_SEG_PATH.resolve()}")


if __name__ == "__main__":
    main()