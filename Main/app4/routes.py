from pathlib import Path

from fastapi import APIRouter, Request, status, UploadFile, File
from fastapi.responses import RedirectResponse, StreamingResponse, JSONResponse
from fastapi.templating import Jinja2Templates

from . import detection

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))


@router.get("/")
def home(request: Request):
    if not request.session.get("is_verified"):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("home4.html", {"request": request})


# ---------------- video streams ----------------
# Two MJPEG streams, matching the two cv2 windows in the standalone
# script: boxes-only and boxes+segmentation. Before any video is
# uploaded these serve a static placeholder frame - no model runs yet.

@router.get("/video_feed_boxes")
def video_feed_boxes():
    return StreamingResponse(
        detection.service.stream_boxes(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


@router.get("/video_feed_seg")
def video_feed_seg():
    return StreamingResponse(
        detection.service.stream_seg(),
        media_type="multipart/x-mixed-replace; boundary=frame",
    )


# ---------------- upload / sync status ----------------

@router.post("/upload_video")
async def upload_video(file: UploadFile = File(...)):
    contents = await file.read()
    result = detection.service.load_video(contents, file.filename)
    if not result.get("ok"):
        return JSONResponse(status_code=400, content={"error": result.get("error", "upload failed")})
    return JSONResponse(content=result)


@router.get("/status")
def get_status():
    return JSONResponse(content=detection.service.get_status())


# ---------------- logs + chat alerts ----------------

@router.get("/logs")
def get_logs(limit: int = 200):
    return JSONResponse(content={"logs": detection.service.get_logs(limit)})


@router.get("/alerts")
def get_alerts(since_id: int = 0):
    return JSONResponse(content={"alerts": detection.service.get_alerts(since_id)})