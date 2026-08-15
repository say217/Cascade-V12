import uuid
from pathlib import Path

from fastapi import APIRouter, File, Request, UploadFile, status
from fastapi.responses import JSONResponse, RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from . import detection

ALLOWED_VIDEO_EXT = {".mp4", ".mov", ".avi", ".mkv", ".webm"}
MAX_UPLOAD_BYTES = 500 * 1024 * 1024  # 500MB safety cap
UPLOAD_CHUNK_BYTES = 1024 * 1024

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


def _is_verified(request: Request) -> bool:
    return bool(request.session.get("is_verified"))


@router.get("/")
def home(request: Request):
    if not _is_verified(request):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("home3.html", {"request": request})


@router.get("/video_feed_boxes")
def video_feed_boxes(request: Request):
    """MJPEG stream: raw frame + bounding boxes only."""
    if not _is_verified(request):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return StreamingResponse(
        detection.mjpeg_generator(stream="boxes"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/video_feed_seg")
def video_feed_seg(request: Request):
    """MJPEG stream: raw frame + bounding boxes + segmentation mask overlay."""
    if not _is_verified(request):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return StreamingResponse(
        detection.mjpeg_generator(stream="seg"),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/logs")
def logs(request: Request, limit: int = 200):
    """Returns the most recent lines from the flood_seg_logs file as JSON,
    used by the live log panel on the dashboard to poll for updates."""
    if not _is_verified(request):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return JSONResponse({"logs": detection.get_recent_logs(limit=limit)})


@router.post("/upload_video")
async def upload_video(request: Request, file: UploadFile = File(...)):
    """Saves an uploaded video to disk, then swaps it in as the active
    detection source. The model is already loaded/warm, so this just
    points the existing playback + inference threads at the new file -
    detections resume in sync with the new video from its first frame."""
    if not _is_verified(request):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)

    ext = Path(file.filename or "").suffix.lower()
    if ext not in ALLOWED_VIDEO_EXT:
        return JSONResponse(
            {"error": f"Unsupported file type: {ext or 'unknown'}. "
                      f"Allowed: {', '.join(sorted(ALLOWED_VIDEO_EXT))}"},
            status_code=400,
        )

    dest_path = Path(detection.UPLOAD_DIR) / f"{uuid.uuid4().hex}{ext}"

    size = 0
    try:
        with open(dest_path, "wb") as out:
            while chunk := await file.read(UPLOAD_CHUNK_BYTES):
                size += len(chunk)
                if size > MAX_UPLOAD_BYTES:
                    raise ValueError("File too large (max 500MB)")
                out.write(chunk)
    except ValueError as exc:
        dest_path.unlink(missing_ok=True)
        return JSONResponse({"error": str(exc)}, status_code=400)

    try:
        detection.load_video(str(dest_path))
    except IOError as exc:
        dest_path.unlink(missing_ok=True)
        return JSONResponse({"error": str(exc)}, status_code=400)

    return JSONResponse({"ok": True, "video_name": file.filename})


@router.get("/status")
def video_status(request: Request):
    """Current active-video / loading state, polled by the UI right after
    an upload so it knows when detections are synced and it can drop the
    loading overlay."""
    if not _is_verified(request):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return JSONResponse(detection.get_status())


from . import filter

@router.get("/alerts")
def get_alerts(request: Request, since_id: int = 0):
    """Returns the most recent flood alerts from filter.py"""
    if not _is_verified(request):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return JSONResponse({"alerts": filter.get_recent_alerts(since_id)})