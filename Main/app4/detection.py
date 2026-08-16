

import cv2
import numpy as np
import threading
import time
import logging
from pathlib import Path
from datetime import datetime
from ultralytics import YOLO
BASE_DIR = Path(__file__).resolve().parent
MODEL_PATH = BASE_DIR.parent.parent / ".model" / "Landslide_segment_model.pt"
UPLOAD_DIR = BASE_DIR / "uploads"
UPLOAD_DIR.mkdir(exist_ok=True)

LOG_PATH = BASE_DIR / "landslide_seg_logs.log"

GRID_X, GRID_Y = 2, 2
CONF_THRES = 0.35
IOU_THRES = 0.45
MAX_BOX_AREA_FRAC = 0.25         # ignore any single box bigger than 25% of frame
INFER_IMGSZ = 480                # smaller = faster on CPU, less accurate (try 320-640)
FEED_EVERY_N_FRAMES = 6
STREAM_FPS = 20
JPEG_QUALITY = 80

PLACEHOLDER_TEXT = "Upload a video to start detection"

logger = logging.getLogger("landslide_detection")
logger.setLevel(logging.INFO)
if not logger.handlers:
    _file_handler = logging.FileHandler(LOG_PATH, encoding="utf-8")
    _file_handler.setFormatter(logging.Formatter("%(asctime)s %(message)s", "%H:%M:%S"))
    logger.addHandler(_file_handler)


def _placeholder_frame(width=960, height=540, text=PLACEHOLDER_TEXT):
    frame = np.zeros((height, width, 3), dtype=np.uint8)
    frame[:] = (13, 13, 13)
    (tw, th), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, 0.8, 2)
    cv2.putText(frame, text, ((width - tw) // 2, (height + th) // 2),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (120, 120, 120), 2, cv2.LINE_AA)
    return frame


def _encode_jpeg(frame):
    ok, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, JPEG_QUALITY])
    return buf.tobytes() if ok else None


class LandslideDetectionService:

    def __init__(self):
        self._lock = threading.Lock()
        self.model = None

        self.cap = None
        self.video_name = None
        self.width = 0
        self.height = 0
        self.frame_area = 0
        self.fps = 25

        self._pending_frame = None
        self._latest_boxes = []
        self._latest_masks = []
        self._latest_update_time = None

        self._latest_boxes_frame = _placeholder_frame()
        self._latest_seg_frame = _placeholder_frame()

        self._loading = False
        self._has_video = False
        self._stop_flag = False
        self._frame_idx = 0

        self._alerts = []
        self._next_alert_id = 1
        self._last_alert_time = 0.0

        self._decode_thread = None
        self._infer_thread = None

    # ---------------- status ----------------

    def get_status(self):
        with self._lock:
            return {
                "loading": self._loading,
                "has_video": self._has_video,
                "video_name": self.video_name,
            }

    # ---------------- model ----------------

    def _ensure_model(self):
        if self.model is None:
            logger.info(f"Loading model from {MODEL_PATH}")
            self.model = YOLO(str(MODEL_PATH))

    # ---------------- upload / swap video ----------------

    def load_video(self, file_bytes: bytes, filename: str):
        """Saves the uploaded file, then swaps it in as the active source.
        Stops any previous decode/inference threads first so no stale
        frames from the old video bleed into the new one."""
        with self._lock:
            self._loading = True
            self._has_video = False

        self._stop_threads()

        dest = UPLOAD_DIR / filename
        dest.write_bytes(file_bytes)

        try:
            self._ensure_model()

            cap = cv2.VideoCapture(str(dest))
            if not cap.isOpened():
                raise IOError(f"Could not open video: {dest}")

            with self._lock:
                self.cap = cap
                self.video_name = filename
                self.width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
                self.height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
                self.frame_area = self.width * self.height
                self.fps = cap.get(cv2.CAP_PROP_FPS) or 25
                self._frame_idx = 0
                self._pending_frame = None
                self._latest_boxes = []
                self._latest_masks = []
                self._latest_update_time = None
                self._stop_flag = False

            logger.info(f"Video loaded: {filename} ({self.width}x{self.height} @ {self.fps:.1f}fps)")

            self._decode_thread = threading.Thread(target=self._decode_loop, daemon=True)
            self._infer_thread = threading.Thread(target=self._inference_loop, daemon=True)
            self._decode_thread.start()
            self._infer_thread.start()

            with self._lock:
                self._loading = False
                self._has_video = True

            return {"ok": True, "video_name": filename}

        except Exception as exc:
            logger.error(f"ERROR loading video {filename}: {exc}")
            with self._lock:
                self._loading = False
                self._has_video = False
            return {"ok": False, "error": str(exc)}

    def _stop_threads(self):
        self._stop_flag = True
        if self._decode_thread and self._decode_thread.is_alive():
            self._decode_thread.join(timeout=1)
        if self._infer_thread and self._infer_thread.is_alive():
            self._infer_thread.join(timeout=1)
        with self._lock:
            if self.cap is not None:
                self.cap.release()
                self.cap = None

    # ---------------- background loops ----------------

    def _decode_loop(self):
   
        delay = max(1.0 / self.fps, 1.0 / STREAM_FPS)
        while not self._stop_flag:
            with self._lock:
                cap = self.cap
            if cap is None:
                break

            ret, frame = cap.read()
            if not ret:
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
                continue

            self._frame_idx += 1
            if self._frame_idx % FEED_EVERY_N_FRAMES == 0:
                with self._lock:
                    self._pending_frame = frame.copy()

            with self._lock:
                boxes, masks = self._latest_boxes, self._latest_masks

            boxes_frame = self._draw_boxes(frame, boxes)
            seg_frame = self._draw_masks(boxes_frame, masks)

            with self._lock:
                self._latest_boxes_frame = boxes_frame
                self._latest_seg_frame = seg_frame

            time.sleep(delay)

    def _inference_loop(self):
        while not self._stop_flag:
            with self._lock:
                frame = self._pending_frame
                self._pending_frame = None

            if frame is None:
                time.sleep(0.005)
                continue

            boxes, masks = self._predict_tiles(frame)

            with self._lock:
                self._latest_boxes = boxes
                self._latest_masks = masks
                self._latest_update_time = time.time()

            self._log_detection(boxes)
            self._maybe_alert(boxes)

    # ---------------- inference (same logic as the standalone script) ----------------

    def _predict_tiles(self, frame):
        """
        Slices the frame into a grid (e.g. 2x2 = 4 tiles) to detect smaller,
        localized landslide pockets instead of one massive box over the
        entire screen.
        """
        tile_h, tile_w = self.height // GRID_Y, self.width // GRID_X
        all_boxes = []
        all_masks = []

        for gy in range(GRID_Y):
            for gx in range(GRID_X):
                x_off, y_off = gx * tile_w, gy * tile_h
                tile = frame[y_off:y_off + tile_h, x_off:x_off + tile_w]

                results = self.model.predict(source=tile, imgsz=INFER_IMGSZ, conf=CONF_THRES,
                                              iou=IOU_THRES, verbose=False)

                if results[0].boxes is not None and len(results[0].boxes) > 0:
                    for box in results[0].boxes:
                        x1, y1, x2, y2 = map(int, box.xyxy[0])
                        conf = float(box.conf[0])

                        global_box = [x1 + x_off, y1 + y_off, x2 + x_off, y2 + y_off, conf]

                        box_w, box_h = (x2 - x1), (y2 - y1)
                        if (box_w * box_h) < (MAX_BOX_AREA_FRAC * self.frame_area):
                            all_boxes.append(global_box)

                if results[0].masks is not None:
                    for mask in results[0].masks.data.cpu().numpy():
                        full_mask = np.zeros((self.height, self.width), dtype=bool)
                        resized_mask = cv2.resize(mask, (tile_w, tile_h)) > 0.5
                        full_mask[y_off:y_off + tile_h, x_off:x_off + tile_w] = resized_mask
                        all_masks.append(full_mask)

        return all_boxes, all_masks

    @staticmethod
    def _draw_boxes(frame, boxes):
        frame = frame.copy()
        for box in boxes:
            x1, y1, x2, y2, conf = box

            cv2.rectangle(frame, (x1, y1), (x2, y2), (255, 255, 255), thickness=1)

            label_text = f"landslide {conf * 100:.1f}%"
            (text_w, text_h), baseline = cv2.getTextSize(label_text, cv2.FONT_HERSHEY_SIMPLEX, 0.35, 1)

            text_x = x1 + 2
            text_y = y1 - 4 if y1 - 4 > 12 else y1 + text_h + 4

            cv2.rectangle(frame, (x1, text_y - text_h - 2), (text_x + text_w + 35, text_y + baseline), (28, 28, 28), -1)
            cv2.putText(frame, label_text, (text_x, text_y), cv2.FONT_HERSHEY_SIMPLEX, 0.35, (255, 255, 255), 1, cv2.LINE_AA)

            bar_x_start = text_x + text_w + 4
            bar_max_width = 25
            bar_width = int(bar_max_width * conf)
            cv2.rectangle(frame, (bar_x_start, text_y - text_h + 2), (bar_x_start + bar_max_width, text_y - text_h + 7), (85, 85, 85), -1)
            cv2.rectangle(frame, (bar_x_start, text_y - text_h + 2), (bar_x_start + bar_width, text_y - text_h + 7), (255, 255, 255), -1)

        return frame

    @staticmethod
    def _draw_masks(frame, masks):
        if not masks:
            return frame.copy()
        frame = frame.copy()
        overlay = frame.copy()
        mask_color = np.array([0, 69, 255], dtype=np.uint8)  # BGR red-orange (landslide/earth tone)
        for mask in masks:
            overlay[mask] = mask_color
        cv2.addWeighted(overlay, 0.35, frame, 0.65, 0, dst=frame)
        return frame

    # ---------------- logging + alerts ----------------

    def _log_detection(self, boxes):
        if boxes:
            confs = ", ".join(f"{b[4] * 100:.1f}%" for b in boxes)
            logger.info(f"Detection: {len(boxes)} landslide region(s) [{confs}]")
        else:
            logger.info("no landslide detected")

    def _maybe_alert(self, boxes):
        # Rate-limit chat alerts (max ~1 per 8s) so the panel doesn't spam
        # a message on every single inference pass.
        now = time.time()
        if not boxes or (now - self._last_alert_time) < 8:
            return
        self._last_alert_time = now
        top_conf = max(b[4] for b in boxes) * 100
        message = (f"Detected {len(boxes)} possible landslide region(s) in the current frame "
                   f"(highest confidence {top_conf:.1f}%).")
        with self._lock:
            alert = {
                "id": self._next_alert_id,
                "message": message,
                "timestamp": datetime.now().isoformat(timespec="seconds"),
            }
            self._next_alert_id += 1
            self._alerts.append(alert)
            self._alerts = self._alerts[-200:]  # cap memory use

    def get_logs(self, limit=200):
        if not LOG_PATH.exists():
            return []
        lines = LOG_PATH.read_text(encoding="utf-8", errors="ignore").splitlines()
        return lines[-limit:]

    def get_alerts(self, since_id=0):
        with self._lock:
            return [a for a in self._alerts if a["id"] > since_id]

    # ---------------- MJPEG streams ----------------

    def stream_boxes(self):
        yield from self._mjpeg_stream(lambda: self._latest_boxes_frame)

    def stream_seg(self):
        yield from self._mjpeg_stream(lambda: self._latest_seg_frame)

    def _mjpeg_stream(self, get_frame):
        delay = 1.0 / STREAM_FPS
        while True:
            with self._lock:
                frame = get_frame()
            payload = _encode_jpeg(frame)
            if payload is not None:
                yield (b"--frame\r\n"
                       b"Content-Type: image/jpeg\r\n\r\n" + payload + b"\r\n")
            time.sleep(delay)
service = LandslideDetectionService()

