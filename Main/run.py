import os
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from fastapi.staticfiles import StaticFiles
from starlette.middleware.sessions import SessionMiddleware

from .app1.routes import router as app1_router
from .app2.routes import router as app2_router
from .app3.routes import router as app3_router
from .app3 import detection as app3_detection
from .app4.routes import router as app4_router

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

PROJECT_ROOT = Path(__file__).resolve().parent.parent
ASSETS_DIR = PROJECT_ROOT / "Assets"
STATIC_DIR = PROJECT_ROOT / "static"

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "change-me"))

app.mount("/Assets", StaticFiles(directory=str(ASSETS_DIR)), name="assets")
app.mount("/static", StaticFiles(directory=str(STATIC_DIR)), name="static")

# Include routers
# NOTE: app3's /video_feed_boxes and /video_feed_seg endpoints live under the
# /app3 prefix (e.g. /app3/video_feed_boxes), same as every other app3 route,
# so there is no collision with /Assets, /static, or the other app prefixes.
app.include_router(app1_router, prefix="/app1")
app.include_router(app2_router, prefix="/app2")
app.include_router(app3_router, prefix="/app3")
app.include_router(app4_router, prefix="/app4")


@app.on_event("shutdown")
def _shutdown_app3_detection():
    # Cleanly stop the YOLO capture/inference threads and release the
    # video capture handle when the server shuts down.
    app3_detection.stop_detection()


@app.get("/")
def root():
    return RedirectResponse(url="/app2/login")