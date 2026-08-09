import os
import secrets
import smtplib
from email.message import EmailMessage

try:
    from dotenv import load_dotenv

    load_dotenv()
except ImportError:
    pass


def get_env_int(name: str, default: int) -> int:
    value = os.getenv(name)
    if value is None or not value.strip():
        return default
    return int(value)


# ---------------------------------------------------------------------------
# Config
# ---------------------------------------------------------------------------
SMTP_HOST = os.getenv("SMTP_HOST", "smtp.gmail.com")
SMTP_PORT = get_env_int("SMTP_PORT", 587)
SMTP_USER = os.getenv("SMTP_USER", "")
SMTP_PASSWORD = os.getenv("SMTP_PASSWORD", "")
SMTP_SENDER = os.getenv("SMTP_SENDER", "") or SMTP_USER
SMTP_USE_TLS = os.getenv("SMTP_USE_TLS", "true").lower() == "true"
SMTP_USE_SSL = os.getenv("SMTP_USE_SSL", "false").lower() == "true"
BASE_URL = os.getenv("BASE_URL", "http://127.0.0.1:8000")
VERIFY_TOKEN_TTL_MINUTES = get_env_int("VERIFY_CODE_EXP_MINUTES", 10)


# ---------------------------------------------------------------------------
# Email (Gmail SMTP + App Password, but works with any SMTP-compatible host)
# ---------------------------------------------------------------------------
def send_verification_email(recipient_email: str, code: str) -> str | None:
    """Returns None on success, or an error message string on failure."""
    if not SMTP_USER or not SMTP_PASSWORD:
        return "SMTP credentials are missing. Set SMTP_USER and SMTP_PASSWORD."

    message = EmailMessage()
    message["Subject"] = "Verify your account"
    message["From"] = SMTP_SENDER
    message["To"] = recipient_email
    message.set_content(
        "Use the verification code below to verify your account:\n\n"
        f"{code}\n\n"
        f"This code expires in {VERIFY_TOKEN_TTL_MINUTES} minutes.\n"
        "If you did not sign up, you can ignore this email."
    )

    try:
        if SMTP_USE_SSL:
            with smtplib.SMTP_SSL(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
        else:
            with smtplib.SMTP(SMTP_HOST, SMTP_PORT, timeout=15) as server:
                server.ehlo()
                if SMTP_USE_TLS:
                    server.starttls()
                    server.ehlo()
                server.login(SMTP_USER, SMTP_PASSWORD)
                server.send_message(message)
    except smtplib.SMTPAuthenticationError:
        return (
            "SMTP authentication failed. If you're using Gmail, make sure you're using a "
            "16-character App Password (not your normal Gmail password) and that "
            "2-Step Verification is enabled on the account."
        )
    except smtplib.SMTPException as exc:
        return f"Failed to send email: {exc}"
    except OSError as exc:
        return f"Could not connect to SMTP server: {exc}"

    return None


def generate_code() -> str:
    return f"{secrets.randbelow(1000000):06d}"