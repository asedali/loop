"""
Transactional email, deliberately thin.

Two backends, chosen by `MAIL_BACKEND`:

    console   print to stdout. The default, and what dev and the whole test
              suite use. It is the reason the password-reset flow is testable
              end to end without a provider or a network: the test reads the link
              out of the captured stdout.
    smtp      smtplib, configured from the environment.

Two rules, both of which exist because of what this app sends:

1. **`send()` never raises into a request.** A provider outage must not turn the
   signup page into a 500, and it must not be reported to the user either — the
   forgot-password response is deliberately identical for a registered and an
   unregistered address, so the user cannot tell a delivery failure from a typo.
   Callers get a bool and log; nothing else changes.
2. **No provider response text reaches the user.** SMTP servers quote the
   recipient address back in error messages, and this app's addresses are not
   published anywhere the user should have to reason about. Failures go to the
   server log (invariant 9).
"""
import smtplib
import ssl
from email.message import EmailMessage

from . import config


def _from() -> str:
    return config.mail_from()


def _console(to: str, subject: str, body: str) -> None:
    """Print the whole message, links included.

    Deliberately prints the live token: this backend exists for development and
    tests, where being unable to click the link would make the flow untestable.
    It must never be the production backend — `send()` says so out loud.
    """
    print(
        f"\n{'=' * 72}\n"
        f"[mailer:console] to={to}\n"
        f"[mailer:console] from={_from()}\n"
        f"[mailer:console] subject={subject}\n"
        f"{'-' * 72}\n{body}\n"
        f"{'=' * 72}\n"
    )


def _smtp(to: str, subject: str, body: str) -> None:
    message = EmailMessage()
    message["From"] = _from()
    message["To"] = to
    message["Subject"] = subject
    message.set_content(body)

    host = config.smtp_host()
    port = config.smtp_port()
    with smtplib.SMTP(host, port, timeout=config.smtp_timeout()) as server:
        if config.smtp_starttls():
            server.starttls(context=ssl.create_default_context())
        username = config.smtp_username()
        if username:
            server.login(username, config.smtp_password())
        server.send_message(message)


def send(to: str, subject: str, body: str) -> bool:
    """Best-effort send. Returns whether it went out; never raises."""
    backend = config.mail_backend()
    try:
        if backend == "smtp":
            _smtp(to, subject, body)
        else:
            # Unknown values degrade to console rather than failing closed: a typo
            # in a deploy config must not silently stop password resets working.
            if backend not in ("console",):
                print(f"[mailer] unknown MAIL_BACKEND '{backend}', using console")
            _console(to, subject, body)
        return True
    except Exception as exc:
        # The recipient address and the provider's error text stay in the server
        # log, never in a response body.
        print(f"[mailer] failed to send '{subject}' to a user via {backend}: {exc!r}")
        return False


# ---------------- the two messages ----------------

def send_email_verification(to: str, token: str) -> bool:
    link = f"{config.app_base_url()}/verify-email?token={token}"
    return send(
        to,
        "Confirm your LaunchLoop email address",
        f"Hi,\n\n"
        f"Confirm this address so we can send you a password reset if you need "
        f"one:\n\n{link}\n\n"
        f"The link works once and expires in "
        f"{config.password_reset_token_minutes()} minutes. If you did not sign "
        f"up, ignore this and nothing happens.\n",
    )


def send_password_reset(to: str, token: str) -> bool:
    link = f"{config.app_base_url()}/reset-password?token={token}"
    return send(
        to,
        "Reset your LaunchLoop password",
        f"Hi,\n\n"
        f"Use this link to choose a new password:\n\n{link}\n\n"
        f"The link works once and expires in "
        f"{config.password_reset_token_minutes()} minutes. If you did not ask for "
        f"this, nothing has changed and you can ignore it.\n",
    )