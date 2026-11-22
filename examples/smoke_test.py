#!/usr/bin/env python3
"""Smoke-test the Ardisa console, desktop logic, and web integration examples."""
from pathlib import Path
import json
import socket
import subprocess
import sys
import time
from urllib.error import URLError
from urllib.request import urlopen

ROOT = Path(__file__).resolve().parents[1]

def run_cli(*args: str) -> str:
    result = subprocess.run(
        ["cargo", "run", "--quiet", "-p", "ardisa", "--", *args],
        cwd=ROOT, capture_output=True, text=True, timeout=120,
    )
    if result.returncode:
        raise AssertionError(result.stderr or f"Ardisa command failed: {args}")
    return result.stdout.strip()

def main() -> None:
    samples = {
        "examples/console.ardisa": "Hello from Ardisa",
        "examples/desktop_app.ardisa": "Hello from the Ardisa desktop application",
        "examples/web_app.ardisa": "200",
    }
    for source, expected in samples.items():
        run_cli("check", source)
        actual = run_cli("run", source)
        if actual != expected:
            raise AssertionError(f"{source}: expected {expected!r}, got {actual!r}")
        print(f"PASS {source}")

    subprocess.run(
        [sys.executable, "-m", "py_compile",
         "examples/desktop-app/app.py", "examples/web-app/server.py"],
        cwd=ROOT, check=True,
    )

    server = subprocess.Popen(
        [sys.executable, "examples/web-app/server.py"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        payload = None
        last_error = None
        for _ in range(30):
            if server.poll() is not None:
                raise AssertionError("Web demo server exited before becoming ready")
            try:
                with urlopen("http://127.0.0.1:8080/api/status", timeout=5) as response:
                    payload = json.loads(response.read().decode("utf-8"))
                break
            except (URLError, TimeoutError, ConnectionError) as error:
                last_error = error
                time.sleep(1)
        if payload is None:
            raise AssertionError(f"Web demo did not become ready: {last_error}")
        if payload.get("ok") is not True or payload.get("status") != 200:
            raise AssertionError(f"Unexpected web response: {payload!r}")
        print("PASS web HTTP integration")
    finally:
        server.terminate()
        try:
            server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            server.kill()
            server.wait(timeout=5)

    with socket.socket() as probe:
        probe.bind(("127.0.0.1", 0))
        port = probe.getsockname()[1]
    native_server = subprocess.Popen(
        ["cargo", "run", "--quiet", "-p", "ardisa", "--", "serve",
         "examples/shared_app_logic.ardisa", "--allow-http-listen",
         f"127.0.0.1:{port}", "GET", "/", "home", "GET", "/echo", "echo_query"],
        cwd=ROOT, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL,
    )
    try:
        body = None
        last_error = None
        for _ in range(30):
            if native_server.poll() is not None:
                raise AssertionError("Native Ardisa HTTP server exited before becoming ready")
            try:
                with urlopen(f"http://127.0.0.1:{port}/", timeout=5) as response:
                    home_body = response.read().decode("utf-8")
                with urlopen(f"http://127.0.0.1:{port}/echo?q=smoke", timeout=5) as response:
                    query_body = response.read().decode("utf-8")
                body = home_body + "|" + query_body
                break
            except (URLError, TimeoutError, ConnectionError) as error:
                last_error = error
                time.sleep(1)
        if body != "Hello from shared Ardisa application logic|q=smoke":
            raise AssertionError(f"Native Ardisa HTTP response mismatch: {body!r}; last error: {last_error}")
        print("PASS native Ardisa HTTP server")
    finally:
        native_server.terminate()
        try:
            native_server.wait(timeout=10)
        except subprocess.TimeoutExpired:
            native_server.kill()
            native_server.wait(timeout=5)

if __name__ == "__main__":
    main()
