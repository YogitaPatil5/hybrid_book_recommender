"""
``python -m bookmind`` — start the web app with one command.

    python -m bookmind                 # http://localhost:8000 (next free port if busy)
    python -m bookmind --port 8001     # explicit port
    python -m bookmind --open          # also open the browser
    python -m bookmind --reload        # auto-reload on code changes (development)

Port and host default to BOOKMIND_PORT / BOOKMIND_HOST from ``.env``. Outside
production, a busy port is not an error: the next free port is used and the
address is printed. In production the configured port is used as-is (fail fast).
"""

from __future__ import annotations

import argparse
import logging
import socket
import threading
import webbrowser
from collections.abc import Sequence

import uvicorn

from bookmind.core.config import get_settings
from bookmind.core.logging import setup_logging

log = logging.getLogger("bookmind")
PORT_SEARCH_RANGE = 20


def port_is_free(host: str, port: int) -> bool:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
        except OSError:
            return False
    return True


def find_free_port(host: str, start: int, attempts: int = PORT_SEARCH_RANGE) -> int:
    for port in range(start, min(start + attempts, 65536)):
        if port_is_free(host, port):
            return port
    raise SystemExit(
        f"No free port between {start} and {start + attempts - 1}. "
        "Close the other server or pass --port."
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        prog="python -m bookmind",
        description=__doc__,
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--port", type=int, help="Port (default: BOOKMIND_PORT or 8000)")
    parser.add_argument("--host", help="Host (default: BOOKMIND_HOST or 0.0.0.0)")
    parser.add_argument("--open", action="store_true", help="Open the app in the browser")
    parser.add_argument("--reload", action="store_true", help="Auto-reload on code changes")
    return parser


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    settings = get_settings()
    setup_logging(settings)

    host = args.host or settings.host
    requested = args.port or settings.port
    port = requested
    if not settings.is_production and not port_is_free(host, requested):
        port = find_free_port(host, requested + 1)
        log.warning("Port %d is busy — using port %d instead", requested, port)

    url = f"http://localhost:{port}"
    log.info("Starting BookMind — open %s in your browser (Ctrl+C to stop)", url)
    if args.open:
        threading.Timer(3.0, webbrowser.open, args=(url,)).start()

    uvicorn.run(
        "bookmind.main:create_app",
        factory=True,
        host=host,
        port=port,
        reload=args.reload or settings.debug,
        log_config=None,  # our own logging config is applied in create_app
        proxy_headers=True,
        access_log=False,  # RequestContextMiddleware writes richer access logs
    )


if __name__ == "__main__":  # pragma: no cover
    main()
