"""Keep secrets out of logs (AGENTS.md rule 7).

Every secret MLPilot reads or stores is registered here (``register_secret``). A log-record
factory then replaces those values, and any ``scheme://user:password@host`` URL, in every
record's message and traceback, whichever logger or handler produces it. A factory is used
instead of a handler filter because it also covers handlers added later, such as a test's.

Secrets shorter than ``MIN_LENGTH`` are not replaced: a one-letter "password" would garble
every log line that contains that letter.
"""

from __future__ import annotations

import logging
import re
import threading

MIN_LENGTH = 4
MASK = "***"

_URL_PASSWORD = re.compile(r"(\b[a-zA-Z][a-zA-Z0-9+.\-]*://[^\s:/@]*:)([^\s@/]+)(@)")
_lock = threading.Lock()
_secrets: tuple[str, ...] = ()
_installed = False


def register_secret(value: str | None) -> None:
    """Remember ``value`` so it is replaced in logs from now on."""
    global _secrets
    if not value or len(value) < MIN_LENGTH:
        return
    with _lock:
        if value not in _secrets:
            # Longest first, so a secret that contains another is replaced whole.
            _secrets = tuple(sorted((*_secrets, value), key=len, reverse=True))


def redact(text: str) -> str:
    """``text`` with every registered secret and URL password masked."""
    for secret in _secrets:
        text = text.replace(secret, MASK)
    return _URL_PASSWORD.sub(rf"\1{MASK}\3", text)


def install() -> None:
    """Wrap the log-record factory once. Safe to call repeatedly."""
    global _installed
    with _lock:
        if _installed:
            return
        _installed = True
    previous = logging.getLogRecordFactory()

    def factory(*args: object, **kwargs: object) -> logging.LogRecord:
        record = previous(*args, **kwargs)  # type: ignore[arg-type]
        try:
            message = record.getMessage()
        except (TypeError, ValueError):
            return record  # a malformed log call; the logging module reports it itself
        cleaned = redact(message)
        if cleaned != message:
            record.msg, record.args = cleaned, ()
        if record.exc_info and record.exc_info[0] is not None:
            formatted = logging.Formatter().formatException(record.exc_info)
            cleaned_exc = redact(formatted)
            if cleaned_exc != formatted:
                # Handlers print exc_text when it is set; drop exc_info so a pretty-printing
                # handler can't rebuild the unredacted traceback from it.
                record.exc_text, record.exc_info = cleaned_exc, None
        return record

    logging.setLogRecordFactory(factory)
