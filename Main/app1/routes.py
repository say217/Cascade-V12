from pathlib import Path

from fastapi import APIRouter, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

@router.get("/")
def home(request: Request):
    if not request.session.get("is_verified"):
        return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)
    return templates.TemplateResponse("home.html", {"request": request})