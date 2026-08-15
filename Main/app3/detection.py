"""
Flood detection engine for App 3.

This mirrors your original standalone script's model/video protocol as
closely as possible - same ROOT_DIR/model_path setup, same predict_tiles /
draw_boxes / draw_masks functions, same single-lock inference_worker
background thread. The two cv2.imshow windows are swapped for two
in-memory JPEG buffers that FastAPI streams to the browser instead (a
server has no display to imshow onto).

Startup behavior: unlike the original script, NO video is opened or
played automatically on import. The YOLO model loads once and stays
warm, but both streams sit idle showing a placeholder frame until the
user uploads a video from the UI, which calls load_video() below. This
avoids ever showing detections for a video the user didn't ask for.

Logging: all status / detection messages go to a rolling log file
(flood_seg_logs) instead of the terminal. The file is capped at
LOG_MAX_LINES lines - once it grows past that, only the most recent
LOG_MAX_LINES lines are kept.
"""

import os
import threading
import time

os.environ["YOLO_VERBOSE"] = "False"
import logging as pylogging
pylogging.getLogger("ultralytics").setLevel(pylogging.ERROR)

import cv2
import numpy as np
from ultralytics import YOLO

from . import filter


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
model_path = os.path.join(ROOT_DIR, '.model', 'flood segment_model_.pt')

video_path = os.path.join(
    ROOT_DIR,
    'Assets',
    'test_vedio',
    'vidssave.com Drone footage of flood damage in Aceh as Indonesia steps up response _ AFP 720P.mp4',
)

GRID_X, GRID_Y = 2, 2            # tiling grid (try 1,1 if 2x2 is still too slow - big speedup)
CONF_THRES = 0.35
IOU_THRES = 0.45
MAX_BOX_AREA_FRAC = 0.25         # ignore any single box bigger than 25% of frame
INFER_IMGSZ = 480                # smaller = faster on CPU, less accurate (try 320-640)
FEED_EVERY_N_FRAMES = 6
JPEG_QUALITY = 80                # quality of the frames streamed to the browser

BOX_COLOR = (0, 69, 255)         # BGR - reddish orange bounding box outline
BAR_FILL_COLOR = (0, 69, 255)    # BGR - reddish orange confidence-bar fill
MASK_COLOR = np.array([255, 165, 0], dtype=np.uint8)  # BGR orange segmentation overlay


LOG_FILE = os.path.join(ROOT_DIR, 'flood_seg_logs')
LOG_MAX_LINES = 2000
LOG_TRIM_EVERY = 20               # only re-check/trim file size every N writes (cheaper than every write)

_log_lock = threading.Lock()
_log_write_count = 0

def clear_logs():
    """Clear all logs from the log file."""
    with _log_lock:
        if os.path.exists(LOG_FILE):
            try:
                os.remove(LOG_FILE)
            except OSError:
                pass

clear_logs()


def _write_log(message: str):
    """Append a timestamped line to the log file. Never raises - logging
    failures must not take down detection/playback."""
    global _log_write_count
    timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
    line = f"[{timestamp}] {message}\n"
    try:
        with _log_lock:
            with open(LOG_FILE, "a", encoding="utf-8") as f:
                f.write(line)
            _log_write_count += 1
            if _log_write_count >= LOG_TRIM_EVERY:
                _log_write_count = 0
                _trim_log_file()
    except OSError:
        pass


def _trim_log_file():
    """Keep only the most recent LOG_MAX_LINES lines in the log file.
    Must be called while holding _log_lock."""
    try:
        with open(LOG_FILE, "r", encoding="utf-8") as f:
            lines = f.readlines()
        if len(lines) > LOG_MAX_LINES:
            with open(LOG_FILE, "w", encoding="utf-8") as f:
                f.writelines(lines[-LOG_MAX_LINES:])
    except OSError:
        pass


def get_recent_logs(limit: int = 200):
    """Return up to the last `limit` lines from the log file, oldest first.
    Used by the /app3/logs endpoint to feed the live log panel in the UI.
    Thread-safe; never raises - returns [] on any read failure."""
    limit = max(1, min(limit, LOG_MAX_LINES))
    try:
        with _log_lock:
            if not os.path.exists(LOG_FILE):
                return []
            with open(LOG_FILE, "r", encoding="utf-8") as f:
                lines = f.readlines()
        return [line.rstrip("\n") for line in lines[-limit:]]
    except OSError:
        return []


_write_log("Loading model...")
model = YOLO(model_path)
_write_log(f"Model loaded from {model_path}")


UPLOAD_DIR = os.path.join(ROOT_DIR, 'Assets', 'uploads')
os.makedirs(UPLOAD_DIR, exist_ok=True)

cap = None                      # no video loaded yet - set by load_video() on first upload
fps = 25
width, height = 1280, 720       # placeholder geometry, used only for the "upload a video" frame
frame_area = width * height
frame_delay = 1 / fps

_video_lock = threading.Lock()
_loading_video = False          # True while an uploaded video is being opened/synced
_current_video_name = None      # set once a video has actually been loaded
_frame_idx = 0                  # module-level so load_video() can reset playback to frame 0


def _build_placeholder_frame(text: str = "Upload a video to start detection"):
    """A plain dark frame with centered text, shown on both streams until
    the user uploads a video (and briefly again if a cap ever needs to be
    swapped and produces no frames)."""
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (18, 18, 18)  # BGR - matches the dashboard's dark theme
    font, scale, thickness = cv2.FONT_HERSHEY_SIMPLEX, 1.0, 2
    (text_w, text_h), _ = cv2.getTextSize(text, font, scale, thickness)
    x, y = (width - text_w) // 2, (height + text_h) // 2
    cv2.putText(frame, text, (x, y), font, scale, (140, 140, 140), thickness, cv2.LINE_AA)
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def predict_tiles(frame):
    """
    Slices the frame into a grid (e.g. 2x2 = 4 tiles) to detect smaller,
    localized flood pockets instead of one massive box over the entire screen.
    """
    tile_h, tile_w = height // GRID_Y, width // GRID_X
    all_boxes = []
    all_masks = []

    for gy in range(GRID_Y):
        for gx in range(GRID_X):
            x_off, y_off = gx * tile_w, gy * tile_h
            tile = frame[y_off:y_off + tile_h, x_off:x_off + tile_w]

            results = model.predict(source=tile, imgsz=INFER_IMGSZ, conf=CONF_THRES,
                                     iou=IOU_THRES, verbose=False)

            if results[0].boxes is not None and len(results[0].boxes) > 0:
                for box in results[0].boxes:
                    x1, y1, x2, y2 = map(int, box.xyxy[0])
                    conf = float(box.conf[0])

                    # Convert tile-relative coordinates back to global frame coordinates
                    global_box = [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off, conf]

                    # Filter out any single box that takes up more than 25% of frame space
                    box_w, box_h = (x2 - x1), (y2 - y1)
                    if (box_w * box_h) < (MAX_BOX_AREA_FRAC * frame_area):
                        all_boxes.append(global_box)

            if results[0].masks is not None:
                for mask in results[0].masks.data.cpu().numpy():
                    full_mask = np.zeros((height, width), dtype=bool)
                    resized_mask = cv2.resize(mask, (tile_w, tile_h)) > 0.5
                    full_mask[y_off:y_off + tile_h, x_off:x_off + tile_w] = resized_mask
                    all_masks.append(full_mask)

    return all_boxes, all_masks


def draw_boxes(frame, boxes):
    """Returns a copy of frame with bounding boxes + HUD labels drawn on it."""
    frame = frame.copy()
    for box in boxes:
        x1, y1, x2, y2, conf = box

        cv2.rectangle(frame, (x1, y1), (x2, y2), BOX_COLOR, thickness=1)

        label_text = f"flood {conf * 100:.1f}%"
        (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

        text_x = x1 + 2
        text_y = y1 - 4 if y1 - 4 > 12 else y1 + text_h + 4

        # Background badge
        cv2.rectangle(frame, (x1, text_y - text_h - 2), (text_x + text_w + 35, text_y + baseline), (28, 28, 28), -1)
        cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

        # Confidence bar
        bar_x_start = text_x + text_w + 4
        bar_max_width = 25
        bar_width = int(bar_max_width * conf)
        cv2.rectangle(frame, (bar_x_start, text_y - text_h + 2), (bar_x_start + bar_max_width, text_y - text_h + 7), (85, 85, 85), -1)
        cv2.rectangle(frame, (bar_x_start, text_y - text_h + 2), (bar_x_start + bar_width, text_y - text_h + 7), BAR_FILL_COLOR, -1)

    return frame


def draw_masks(frame, masks):
    """Returns a copy of frame with the segmentation overlay blended on top."""
    if not masks:
        return frame.copy()
    frame = frame.copy()
    overlay = frame.copy()
    for mask in masks:
        overlay[mask] = MASK_COLOR
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
    return frame



_lock = threading.Lock()
_pending_frame = None       # next frame waiting to be picked up by the worker
_latest_boxes = []
_latest_masks = []
_latest_update_time = None
_stop_flag = False


_placeholder_jpeg = _build_placeholder_frame()
_det_jpeg = _placeholder_jpeg
_seg_jpeg = _placeholder_jpeg


def inference_worker():
    global _pending_frame, _latest_boxes, _latest_masks, _latest_update_time
    while not _stop_flag:
        with _lock:
            frame = _pending_frame
            _pending_frame = None  # consume it

        if frame is None:
            time.sleep(0.005)
            continue

        boxes, masks = predict_tiles(frame)

        with _lock:
            _latest_boxes = boxes
            _latest_masks = masks
            _latest_update_time = time.time()
            
        filter.process_detections(boxes)

        # Continuous detection log - one line per inference pass.
        if boxes:
            avg_conf = sum(b[4] for b in boxes) / len(boxes)
            _write_log(
                f"Detection: {len(boxes)} box(es), {len(masks)} mask(s), "
                f"avg_conf={avg_conf:.2f}"
            )
        else:
            _write_log(f"Detection: no flood detected, {len(masks)} mask(s)")


worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()


def _encode(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


def playback_loop():
    global _pending_frame, _det_jpeg, _seg_jpeg, _frame_idx

    while not _stop_flag:
 
        if _loading_video:
            time.sleep(0.02)
            continue

        with _video_lock:
            if cap is None:
                have_video = False
            else:
                have_video = True
                ret, frame = cap.read()
                local_frame_delay = frame_delay

        if not have_video:
            time.sleep(0.05)
            continue

        if not ret:
    
            _write_log("End of video reached - looping back to start")
            with _video_lock:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
            _frame_idx = 0
            continue


        if _frame_idx % FEED_EVERY_N_FRAMES == 0:
            with _lock:
                _pending_frame = frame.copy()

        with _lock:
            boxes, masks, updated_at = _latest_boxes, _latest_masks, _latest_update_time

        det_frame = draw_boxes(frame, boxes)               # window 1: boxes only
        seg_frame = draw_masks(det_frame, masks)            # window 2: boxes + segmentation

        if updated_at is not None:
            age_s = time.time() - updated_at
            cv2.putText(det_frame, f"detections: {age_s:.1f}s old", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        det_jpeg = _encode(det_frame)
        seg_jpeg = _encode(seg_frame)

        with _lock:
            if det_jpeg is not None:
                _det_jpeg = det_jpeg
            if seg_jpeg is not None:
                _seg_jpeg = seg_jpeg

        _frame_idx += 1
        time.sleep(local_frame_delay)


playback = threading.Thread(target=playback_loop, daemon=True)
playback.start()
def load_video(path: str):
    """
    Swap the active video source to `path` (called after a user uploads a
    video). The YOLO model stays loaded/warm - only the cv2.VideoCapture
    and its geometry (fps/width/height) change.

    Synchronization: while the swap is happening, playback_loop() is
    paused (via _loading_video) and any in-flight/stale detections from
    the *previous* video are cleared immediately. This means the moment
    the new video starts streaming, its overlay is empty rather than
    showing leftover boxes drawn for a different clip - detections only
    reappear once the inference worker has actually processed a frame
    from the new video. frame_idx is also reset to 0 so tile feeding
    (every FEED_EVERY_N_FRAMES) restarts cleanly from the new video's
    first frame.
    """
    global cap, fps, width, height, frame_area, frame_delay
    global _loading_video, _pending_frame, _latest_boxes, _latest_masks, _latest_update_time
    global _frame_idx, _current_video_name

    with _video_lock:
        _loading_video = True
        
        clear_logs()
        filter.clear_alerts()
        
        _write_log(f"Loading uploaded video: {os.path.basename(path)}")

        # Clear anything mid-flight so the old video's detections can never
        # be drawn over the new video's frames.
        with _lock:
            _pending_frame = None
            _latest_boxes = []
            _latest_masks = []
            _latest_update_time = None

        new_cap = cv2.VideoCapture(path)
        if not new_cap.isOpened():
            _loading_video = False
            _write_log(f"ERROR: could not open uploaded video: {path}")
            raise IOError(f"Could not open video: {path}")

        old_cap = cap
        cap = new_cap
        if old_cap is not None:
            old_cap.release()

        fps = cap.get(cv2.CAP_PROP_FPS) or 25
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        frame_area = width * height
        frame_delay = 1 / fps if fps else 0.04

        _frame_idx = 0
        _current_video_name = os.path.basename(path)
        _loading_video = False

    _write_log(
        f"Video source switched to {_current_video_name} "
        f"({width}x{height} @ {fps:.1f}fps) - model already warm, playback resumed"
    )


def get_status():
    """Snapshot for the UI: which video is active and whether a swap is
    still in progress (used to show a loading state on the upload button
    and video panels)."""
    return {
        "loading": _loading_video,
        "has_video": cap is not None,
        "video_name": _current_video_name,
        "width": width,
        "height": height,
        "fps": round(fps, 1) if fps else None,
    }


def stop_detection():
    """Call on app shutdown to stop both background threads and release the video."""
    global _stop_flag
    _stop_flag = True
    worker.join(timeout=1)
    playback.join(timeout=1)
    if cap is not None:
        cap.release()
    _write_log("Detection stopped, video released")


def mjpeg_generator(stream: str = "boxes"):
    """
    Yields multipart/x-mixed-replace chunks for either the "boxes" stream
    (WIN_DET equivalent) or the "seg" stream (WIN_SEG equivalent), always
    the freshest frame the playback loop has produced.
    """
    while not _stop_flag:
        with _lock:
            jpeg = _det_jpeg if stream == "boxes" else _seg_jpeg

        if jpeg is not None:
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + jpeg + b"\r\n"
            )

        time.sleep(1 / 30)