#!/usr/bin/env python3
"""Local HTTP adapter executing Ardisa for /api/status.

Run from the repository root: python3 examples/web-app/server.py
Then open http://127.0.0.1:8080
"""
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
import json
import subprocess

ROOT = Path(__file__).resolve().parents[2]
CLI = ["cargo", "run", "--quiet", "-p", "ardisa", "--", "run", "examples/web_app.ardisa"]


class Handler(BaseHTTPRequestHandler):
    def do_GET(self):
        if self.path == "/":
            body = (ROOT / "examples/web-app/index.html").read_bytes()
            self.send_response(200)
            self.send_header("Content-Type", "text/html; charset=utf-8")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)
            return

        if self.path == "/api/status":
            try:
                result = subprocess.run(CLI, cwd=ROOT, check=True, capture_output=True,
                                        text=True, timeout=45)
                status = int(result.stdout.strip())
                payload = json.dumps({"ok": status == 200, "status": status,
                                      "runtime": "Ardisa native execution"}).encode()
                self.send_response(status if 100 <= status <= 599 else 500)
            except (subprocess.SubprocessError, ValueError) as error:
                payload = json.dumps({"ok": False, "error": "Ardisa execution failed",
                                      "detail": str(error)}).encode()
                self.send_response(500)
            self.send_header("Content-Type", "application/json; charset=utf-8")
            self.send_header("Cache-Control", "no-store")
            self.send_header("Content-Length", str(len(payload)))
            self.end_headers()
            self.wfile.write(payload)
            return
        self.send_error(404, "Not found")

    def log_message(self, fmt, *args):
        print("%s - %s" % (self.address_string(), fmt % args))


if __name__ == "__main__":
    server = ThreadingHTTPServer(("127.0.0.1", 8080), Handler)
    print("Ardisa demo running at http://127.0.0.1:8080 (Ctrl+C to stop)")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()
