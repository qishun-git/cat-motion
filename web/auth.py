from __future__ import annotations

import logging
import smtplib
from email.message import EmailMessage
from typing import Optional, Tuple
from urllib.parse import urljoin

from itsdangerous import BadSignature, SignatureExpired, URLSafeTimedSerializer

from cat_motion.config import AuthConfig

logger = logging.getLogger(__name__)

SESSION_SALT = "cat-motion.session"
LOGIN_SALT = "cat-motion.magic-link"


def _normalize_email(email: str) -> str:
    return email.strip().lower()


class MagicLinkAuth:
    def __init__(self, config: AuthConfig) -> None:
        self.config = config
        self._allowed = {_normalize_email(addr) for addr in config.allowed_emails}
        self._login_serializer = URLSafeTimedSerializer(config.secret_key, salt=LOGIN_SALT)
        self._session_serializer = URLSafeTimedSerializer(config.secret_key, salt=SESSION_SALT)

    def is_allowed(self, email: str) -> bool:
        return _normalize_email(email) in self._allowed

    def send_login_link(self, email: str, base_url: str, next_path: Optional[str] = None) -> bool:
        normalized = _normalize_email(email)
        if not self.is_allowed(normalized):
            logger.info("Login requested for non-allowed email: %s", normalized)
            return False
        token = self._create_login_token(normalized, next_path)
        link = self._build_callback_url(base_url, token)
        self._deliver_magic_link(normalized, link)
        logger.info("Magic link sent to %s", normalized)
        return True

    def verify_login_token(self, token: str) -> Tuple[Optional[str], Optional[str]]:
        try:
            data = self._login_serializer.loads(token, max_age=self.config.token_ttl_minutes * 60)
        except (BadSignature, SignatureExpired):
            return None, None
        email = data.get("email")
        next_path = data.get("next")
        if not email or not self.is_allowed(email):
            return None, None
        return email, next_path

    def create_session(self, email: str) -> str:
        normalized = _normalize_email(email)
        if not self.is_allowed(normalized):
            raise ValueError("Email is not allowed")
        return self._session_serializer.dumps({"email": normalized})

    def verify_session(self, raw_token: Optional[str]) -> Optional[str]:
        if not raw_token:
            return None
        try:
            data = self._session_serializer.loads(raw_token, max_age=self.config.session_ttl_hours * 3600)
        except (BadSignature, SignatureExpired):
            return None
        email = data.get("email")
        if not email or not self.is_allowed(email):
            return None
        return email

    def _create_login_token(self, email: str, next_path: Optional[str]) -> str:
        payload = {"email": email}
        if next_path:
            payload["next"] = next_path
        return self._login_serializer.dumps(payload)

    def _build_callback_url(self, base_url: str, token: str) -> str:
        base = base_url.rstrip("/")
        callback_path = f"/auth/callback?token={token}"
        return urljoin(base + "/", callback_path.lstrip("/"))

    def _deliver_magic_link(self, email: str, link: str) -> None:
        mail_cfg = self.config.mail
        message = EmailMessage()
        message["Subject"] = "Your Cat Motion login link"
        message["From"] = mail_cfg.from_email
        message["To"] = email
        body = (
            "Hello,\n\n"
            "Use the one-time link below to access Cat Motion. "
            "This link expires in {minutes} minutes.\n\n"
            "{link}\n\n"
            "If you didn't request this email, you can ignore it.\n"
        ).format(minutes=self.config.token_ttl_minutes, link=link)
        message.set_content(body)
        if mail_cfg.use_ssl and mail_cfg.use_tls:
            raise ValueError("Set either use_tls or use_ssl, not both.")
        if mail_cfg.use_ssl:
            with smtplib.SMTP_SSL(mail_cfg.host, mail_cfg.port, timeout=30) as client:
                self._maybe_login(client, mail_cfg)
                client.send_message(message)
        else:
            with smtplib.SMTP(mail_cfg.host, mail_cfg.port, timeout=30) as client:
                if mail_cfg.use_tls:
                    client.starttls()
                self._maybe_login(client, mail_cfg)
                client.send_message(message)

    @staticmethod
    def _maybe_login(client: smtplib.SMTP, mail_cfg) -> None:
        if mail_cfg.username and mail_cfg.password:
            client.login(mail_cfg.username, mail_cfg.password)
