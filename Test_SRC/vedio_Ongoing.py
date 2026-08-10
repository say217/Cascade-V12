import os
import threading
import time
import cv2
import numpy as np
from ultralytics import YOLO


ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
model_path = os.path.join(ROOT_DIR, '.model', 'flood segment_model_.pt')
video_path = os.path.join(
    ROOT_DIR,
    'Assets',
    'test_vedio',
    'vidssave.com Drone footage of flood damage in Aceh as Indonesia steps up response _ AFP 720P.mp4',
)

# ============================================================
# CONFIG
# ============================================================
GRID_X, GRID_Y = 2, 2            # tiling grid (try 1,1 if 2x2 is still too slow - big speedup)
CONF_THRES = 0.35
IOU_THRES = 0.45
MAX_BOX_AREA_FRAC = 0.25         # ignore any single box bigger than 25% of frame
INFER_IMGSZ = 480                # smaller = faster on CPU, less accurate (try 320-640)
FEED_EVERY_N_FRAMES = 6        
DISPLAY_WIDTH = 700             


print("Loading model...")
model = YOLO(model_path)

print("Opening video...")
cap = cv2.VideoCapture(video_path)
if not cap.isOpened():
    raise IOError(f"Could not open video: {video_path}")

fps = cap.get(cv2.CAP_PROP_FPS) or 25
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
frame_area = width * height
frame_delay_ms = max(1, int(1000 / fps))

scale = DISPLAY_WIDTH / width
disp_w, disp_h = DISPLAY_WIDTH, int(height * scale)

WIN_DET = "Flood Detection - Bounding Boxes"
WIN_SEG = "Flood Detection - Boxes + Segmentation"
cv2.namedWindow(WIN_DET, cv2.WINDOW_NORMAL)
cv2.namedWindow(WIN_SEG, cv2.WINDOW_NORMAL)
cv2.resizeWindow(WIN_DET, disp_w, disp_h)
cv2.resizeWindow(WIN_SEG, disp_w, disp_h)


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

        cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)

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
        cv2.rectangle(frame, (bar_x_start, text_y - text_h + 2), (bar_x_start + bar_width, text_y - text_h + 7), (255, 255, 255), -1)

    return frame


def draw_masks(frame, masks):
    """Returns a copy of frame with the segmentation overlay blended on top."""
    if not masks:
        return frame.copy()
    frame = frame.copy()
    overlay = frame.copy()
    mask_color = np.array([255, 165, 0], dtype=np.uint8)  # BGR orange
    for mask in masks:
        overlay[mask] = mask_color
    cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
    return frame


# ============================================================
# BACKGROUND INFERENCE WORKER
# ============================================================
# The model runs in its own thread so a slow CPU inference pass never
# blocks the video from decoding/displaying the next frame. The worker
# always grabs the MOST RECENT frame handed to it - if it's still busy
# when a newer frame arrives, the older one is simply dropped - so it
# never builds up a backlog, it just gives you the freshest detection
# it can keep up with. This is what actually fixes the stutter: the
# video loop below never waits on the model at all.

_lock = threading.Lock()
_pending_frame = None       # next frame waiting to be picked up by the worker
_latest_boxes = []
_latest_masks = []
_latest_update_time = None
_stop_flag = False


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


worker = threading.Thread(target=inference_worker, daemon=True)
worker.start()


# ============================================================
# MAIN LOOP - live playback, two windows, never blocks on the model
# ============================================================
frame_idx = 0
paused = False

print("Starting playback.")
print("  'q' = quit   |   'p' = pause/resume")

while True:
    if not paused:
        ret, frame = cap.read()
        if not ret:
            print("End of video.")
            break

        # Hand every Nth frame to the background detector thread. We never
        # wait for a result here - the display loop always runs at normal
        # speed no matter how slow the CPU is at inference.
        if frame_idx % FEED_EVERY_N_FRAMES == 0:
            with _lock:
                _pending_frame = frame.copy()

        with _lock:
            boxes, masks, updated_at = _latest_boxes, _latest_masks, _latest_update_time

        det_frame = draw_boxes(frame, boxes)              # window 1: boxes only
        seg_frame = draw_masks(det_frame, masks)           # window 2: boxes + segmentation

        if updated_at is not None:
            age_s = time.time() - updated_at
            cv2.putText(det_frame, f"detections: {age_s:.1f}s old", (10, 20),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1, cv2.LINE_AA)

        cv2.imshow(WIN_DET, cv2.resize(det_frame, (disp_w, disp_h)))
        cv2.imshow(WIN_SEG, cv2.resize(seg_frame, (disp_w, disp_h)))

        frame_idx += 1

    key = cv2.waitKey(1 if paused else frame_delay_ms) & 0xFF
    if key == ord('q'):
        break
    elif key == ord('p'):
        paused = not paused

_stop_flag = True
worker.join(timeout=1)
cap.release()
cv2.destroyAllWindows()