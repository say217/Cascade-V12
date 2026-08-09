from datetime import datetime, timedelta
from pathlib import Path

import bcrypt
from fastapi import APIRouter, Form, Request, status
from fastapi.responses import RedirectResponse
from fastapi.templating import Jinja2Templates

from . import db
from .email_utils import VERIFY_TOKEN_TTL_MINUTES, generate_code, send_verification_email

router = APIRouter()

templates = Jinja2Templates(directory=str(Path(__file__).resolve().parent / "templates"))

@router.get("/")
def home(request: Request):
    return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/signup")
def signup_form(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})


@router.post("/signup")
def signup(
    request: Request,
    email: str = Form(...),
    username: str = Form(...),
    password: str = Form(...),
):
    email = email.strip().lower()
    username = username.strip()

    if db.find_user_by_email_or_username(email, username):
        return templates.TemplateResponse(
            "signup.html",
            {"request": request, "error": "Email or username already exists."},
        )

    password_hash = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt()).decode("utf-8")
    verification_code = generate_code()
    expires_at = (datetime.utcnow() + timedelta(minutes=VERIFY_TOKEN_TTL_MINUTES)).isoformat()

    db.create_user(email, username, password_hash, verification_code, expires_at)

    error = send_verification_email(email, verification_code)
    if error:
        return templates.TemplateResponse(
            "verify.html",
            {"request": request, "error": error, "email": email},
        )

    return templates.TemplateResponse(
        "verify.html",
        {"request": request, "email": email, "message": "Verification code sent."},
    )


@router.get("/verify")
def verify_form(request: Request, email: str | None = None):
    return templates.TemplateResponse(
        "verify.html",
        {"request": request, "email": email},
    )


@router.post("/verify")
def verify_account(request: Request, email: str = Form(...), code: str = Form(...)):
    email = email.strip().lower()

    user = db.get_user_for_verification(email)

    if not user:
        return templates.TemplateResponse(
            "verify.html",
            {"request": request, "error": "Email not found.", "email": email},
        )

    if user["is_verified"]:
        request.session["user_id"] = user["id"]
        request.session["is_verified"] = True
        return RedirectResponse(url="/app1/", status_code=status.HTTP_303_SEE_OTHER)

    if db.is_code_expired(user["code_expires_at"]):
        return templates.TemplateResponse(
            "verify.html",
            {"request": request, "error": "Verification code has expired.", "email": email},
        )

    if user["verification_code"] != code:
        return templates.TemplateResponse(
            "verify.html",
            {"request": request, "error": "Invalid verification code.", "email": email},
        )

    db.mark_user_verified(user["id"])

    request.session["user_id"] = user["id"]
    request.session["is_verified"] = True
    return RedirectResponse(url="/app1/", status_code=status.HTTP_303_SEE_OTHER)


@router.post("/resend-code")
def resend_code(request: Request, email: str = Form(...)):
    email = email.strip().lower()

    user = db.get_user_for_resend(email)

    if not user:
        return templates.TemplateResponse(
            "verify.html",
            {"request": request, "error": "Email not found.", "email": email},
        )

    if user["is_verified"]:
        return templates.TemplateResponse(
            "verify.html",
            {"request": request, "error": "This account is already verified.", "email": email},
        )

    verification_code = generate_code()
    expires_at = (datetime.utcnow() + timedelta(minutes=VERIFY_TOKEN_TTL_MINUTES)).isoformat()
    db.set_new_verification_code(user["id"], verification_code, expires_at)

    error = send_verification_email(email, verification_code)
    if error:
        return templates.TemplateResponse(
            "verify.html",
            {"request": request, "error": error, "email": email},
        )

    return templates.TemplateResponse(
        "verify.html",
        {"request": request, "email": email, "message": "A new verification code has been sent."},
    )


@router.get("/login")
def login_form(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})


@router.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    email = email.strip().lower()

    user = db.get_user_for_login(email)

    if not user:
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password."},
        )

    if not bcrypt.checkpw(password.encode("utf-8"), user["password_hash"].encode("utf-8")):
        return templates.TemplateResponse(
            "login.html",
            {"request": request, "error": "Invalid email or password."},
        )

    if not user["is_verified"]:
        return templates.TemplateResponse(
            "login.html",
            {
                "request": request,
                "error": "Please verify your email before logging in.",
                "email": email,
            },
        )

    request.session["user_id"] = user["id"]
    request.session["is_verified"] = True
    return RedirectResponse(url="/app1/", status_code=status.HTTP_303_SEE_OTHER)


@router.get("/logout")
def logout(request: Request):
    request.session.clear()
    return RedirectResponse(url="/app2/login", status_code=status.HTTP_303_SEE_OTHER)


db.ensure_tables()