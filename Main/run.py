import os

from fastapi import FastAPI
from fastapi.responses import RedirectResponse
from starlette.middleware.sessions import SessionMiddleware

from .app1.routes import router as app1_router
from .app2.routes import router as app2_router
from .app3.routes import router as app3_router
from .app4.routes import router as app4_router

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass

app = FastAPI()
app.add_middleware(SessionMiddleware, secret_key=os.getenv("SECRET_KEY", "change-me"))

# Include routers
app.include_router(app1_router, prefix="/app1")
app.include_router(app2_router, prefix="/app2")
app.include_router(app3_router, prefix="/app3")
app.include_router(app4_router, prefix="/app4")


@app.get("/")
def root():
    return RedirectResponse(url="/app2/login")