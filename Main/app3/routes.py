from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse, StreamingResponse
from fastapi.templating import Jinja2Templates

from . import detection

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