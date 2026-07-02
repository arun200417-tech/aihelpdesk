"""Development launcher for the FastAPI backend — PLAIN HTTP, no TLS.

Binds to 127.0.0.1:8000 and never passes ssl_keyfile/ssl_certfile, so the dev
server is plain HTTP (no certificate issues). If the port is unavailable — in
use (WinError 10048) or access-forbidden (WinError 10013) — it automatically
falls back to the next free port. Do NOT use this for production; enable TLS via
a real reverse proxy / explicit config only when deploying.

Usage (run from the backend/ folder or anywhere):
    python run_dev.py                 # http://127.0.0.1:8000 (auto-fallback if busy)
    python run_dev.py --port 8080     # force a specific port
    python run_dev.py --no-reload     # disable auto-reload
"""
from __future__ import annotations

import argparse
import os
import socket
import sys

DEFAULT_HOST = "127.0.0.1"
DEFAULT_PORT = 8000


def _port_is_free(host: str, port: int) -> bool:
    """True if we can bind host:port. Catches WSAEADDRINUSE (10048) AND
    WSAEACCES (10013, 'socket access forbidden') and treats both as unavailable."""
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as sock:
        try:
            sock.bind((host, port))
            return True
        except OSError:
            return False


def _pick_port(host: str, start: int, tries: int = 25) -> int:
    for port in range(start, start + tries):
        if _port_is_free(host, port):
            return port
    raise SystemExit(f"No free TCP port found in {start}-{start + tries - 1} on {host}.")


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the helpdesk backend for local dev (plain HTTP, TLS disabled).")
    parser.add_argument("--host", default=DEFAULT_HOST)
    parser.add_argument("--port", type=int, default=DEFAULT_PORT)
    parser.add_argument("--no-reload", action="store_true", help="disable auto-reload")
    args = parser.parse_args()

    # Make `app.main:app` importable and keep the working dir at backend/ so the
    # reload subprocess resolves it too. (Data paths are absolute, so this is safe.)
    backend_dir = os.path.dirname(os.path.abspath(__file__))
    sys.path.insert(0, backend_dir)
    os.chdir(backend_dir)

    port = args.port if _port_is_free(args.host, args.port) else _pick_port(args.host, args.port + 1)
    if port != args.port:
        print(f"[run_dev] Port {args.port} unavailable (in use or WinError 10013) -> using {port}.")

    import uvicorn

    print(f"[run_dev] Serving PLAIN HTTP on http://{args.host}:{port}  (no ssl_keyfile/ssl_certfile)")
    uvicorn.run(
        "app.main:app",
        host=args.host,
        port=port,
        reload=not args.no_reload,
        # Intentionally NO ssl_keyfile / ssl_certfile -> TLS disabled for dev.
    )


if __name__ == "__main__":
    main()
