# Project Improvement Guide

Practical, point-form recommendations for security, developer productivity, and runtime smoothness/efficiency, based on the current `detection.py` / `routes.py` / `home4.html` setup.

---

## 1. Security

**Auth coverage**
- Only `GET /` checks `request.session.get("is_verified")`. Every other route (`video_feed_boxes`, `video_feed_seg`, `upload_video`, `status`, `logs`, `alerts`) is currently open to anyone who hits the URL directly.
  - Fix: add a shared dependency and apply it to the whole router.
    ```python
    from fastapi import Depends, HTTPException

    def require_session(request: Request):
        if not request.session.get("is_verified"):
            raise HTTPException(status_code=401, detail="Not authenticated")

    router = APIRouter(dependencies=[Depends(require_session)])
    ```

**File upload hardening**
- No size limit — a huge file will fill disk/RAM. Cap it (e.g. 500 MB) and reject early.
- No extension/content-type allow-list — anything gets written to disk as-is. Restrict to `.mp4 .mov .avi .mkv .webm`.
- Filename is used verbatim (`UPLOAD_DIR / filename`) → path traversal risk (`../../etc/passwd`) and filename collisions across users.
  - Fix: sanitize with `Path(filename).name`, and store under a generated id instead of the raw name:
    ```python
    import uuid
    safe_name = f"{uuid.uuid4().hex}_{Path(filename).name}"
    ```

**Resource exhaustion**
- Nothing stops one user from uploading repeatedly and spinning up new decode/inference threads faster than old ones stop. Add a simple lock/flag ("upload in progress, reject concurrent uploads") or a queue.
- `/logs` and `/alerts` take unbounded `limit`/`since_id` query params — clamp `limit` (e.g. `max(1, min(limit, 1000))`).

**Secrets & config**
- `MODEL_PATH` and other constants are hardcoded in source. Move to environment variables / a `.env` file (via `pydantic-settings` or `python-dotenv`) so paths differ per deployment without code edits.
- Confirm the session middleware's `secret_key` isn't hardcoded in a committed file — load from env.

**Transport & cookies**
- Ensure `SessionMiddleware` (or whatever sets `is_verified`) uses `https_only=True` and `same_site="lax"` in production.
- Serve behind HTTPS; MJPEG over plain HTTP on a public network exposes the video feed and session cookie.

**Dependency hygiene**
- Pin versions in `requirements.txt` and run `pip-audit` or `safety check` in CI — `ultralytics`/`opencv-python` update frequently and occasionally ship CVEs.

---

## 2. Productivity / Developer Workflow

**Config centralization**
- Pull all the `CONF_THRES`, `GRID_X/Y`, `FEED_EVERY_N_FRAMES`, etc. constants into a single `config.py` or `Settings` class so tuning doesn't require hunting through `detection.py`.

**Testing**
- `_predict_tiles`, `_draw_boxes`, `_draw_masks` are pure-ish functions (given a frame + boxes/masks) — easy to unit test with a synthetic `numpy` frame and fake YOLO output, without needing a real model or video file.
- Add a smoke test for the routes using FastAPI's `TestClient` + a tiny sample `.mp4` fixture.

**Linting/formatting**
- Add `ruff` + `black` (or `flake8` + `isort`) and a pre-commit hook so style stays consistent as the app grows across `app1`–`app4`.

**Logging**
- Current logger writes everything to one flat file with no rotation — it'll grow unbounded over a long-running demo. Swap to `logging.handlers.RotatingFileHandler` (e.g. 5 MB × 3 backups).
- Add a `level` distinction: detections at `INFO`, model/video errors at `ERROR` — already partially done, just make sure nothing sensitive (file paths, session data) leaks into log lines.

**Documentation**
- A short `README.md` per app (`app4/README.md`) listing: required model path, expected video formats, env vars, and the route list — saves re-deriving this from source every time.

---

## 3. Performance / Smoothness

**Threading model**
- Current design: 1 decode thread + 1 inference thread per active video, sharing one `_lock`. That's fine for a single demo user but won't scale to multiple concurrent viewers/uploads — each upload tears down and restarts both threads globally (shared `service` singleton). If multi-user support is a goal, key the service by session/video id instead of a single module-level singleton.

**Inference speed**
- `model.predict(...)` runs once per tile (4× per frame at `GRID_X=GRID_Y=2`) — check if `ultralytics` supports batched tile inference (`model.predict(source=[tile1, tile2, tile3, tile4])`) to cut Python-loop overhead and get one batched GPU/CPU pass instead of four sequential ones.
- Confirm device selection is explicit: `model.to("cuda")` if a GPU is available, otherwise CPU inference at `imgsz=480` with 4 tiles per frame can get expensive — consider dropping to `GRID_X=GRID_Y=1` (already flagged as an option) if latency matters more than localizing small regions.
- Model is loaded lazily on first upload — good — but consider a `/warmup` step (predict on a blank frame once at startup) so the *first* real detection isn't slower than the rest.

**Streaming efficiency**
- `_mjpeg_stream` re-encodes JPEG on a fixed timer (`STREAM_FPS`) even if the underlying annotated frame hasn't changed since the last tick — wastes CPU when the source video is paused/looped slowly. Track a "frame changed" flag and skip re-encoding identical frames.
- MJPEG over `multipart/x-mixed-replace` is simple but has real latency/bandwidth cost with multiple viewers (each client gets its own re-encode loop). If you outgrow the demo stage, consider WebSocket or HLS for lower overhead with multiple simultaneous viewers.

**Video looping**
- `_decode_loop` calls `cap.set(cv2.CAP_PROP_POS_FRAMES, 0)` on EOF to loop — this is a known slow/unreliable seek on some codecs (especially with `H.264` B-frames). If looping stutters, re-open the `VideoCapture` from the file path instead of seeking.

**Memory bounds**
- `_alerts` is capped at 200 (good). Logs are read from disk each poll — fine at current scale, but if `landslide_seg_logs.log` grows large, `read_text().splitlines()` re-reads the whole file every 2s from every connected client. Switch to reading only the last N KB (`seek` from the end) once the file gets big.

**Graceful shutdown**
- No FastAPI `@app.on_event("shutdown")` hook to call `detection.service._stop_threads()` — on app restart, the decode/inference threads are daemonized so they die with the process, but the `VideoCapture` handle may not release cleanly. Add an explicit shutdown hook.

---

## 4. Reliability / Error Handling

- Wrap `cap.read()` failures beyond just "EOF" — a corrupted video mid-stream currently just spins in the `while` loop retrying `set(POS_FRAMES, 0)`. Add a retry counter and break/log after N consecutive failures.
- `upload_video` catches exceptions broadly (`except Exception as exc`) and returns the raw exception string to the client — fine for an internal demo, but avoid returning raw exception text in a public-facing deployment (info disclosure). Log the full exception server-side, return a generic message to the client.
- Add a `/health` endpoint (no auth) for uptime checks / load balancers, separate from `/status` (which reports video state, not process health).

---

## Suggested Priority Order

1. **Security**: auth on all routes, upload validation (size/type/filename), no raw exception leakage.
2. **Reliability**: shutdown hook, bounded log reads, corrupted-video handling.
3. **Performance**: batched tile inference, skip redundant JPEG encodes, GPU device check.
4. **Productivity**: config centralization, tests, log rotation, README.