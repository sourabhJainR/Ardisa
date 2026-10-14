# Ardisa web app demo

This runnable integration example uses Python's standard-library HTTP server as a thin adapter; the status decision is an Ardisa program executed by Ardisa's native runner. Ardisa does not yet provide a native HTTP server or browser compiler, and this demo does not claim otherwise.

## Requirements
- Rust toolchain with Cargo
- Python 3.10+
- Run commands from the repository root

## Console example
```sh
cargo run -p ardisa -- run examples/console.ardisa
```
Expected output:
```text
Hello from Ardisa
```

## Web example
```sh
python3 examples/web-app/server.py
```
Open http://127.0.0.1:8080. The page calls `/api/status`; the server executes:
```sh
cargo run --quiet -p ardisa -- run examples/web_app.ardisa
```
Expected API response:
```json
{"ok": true, "status": 200, "runtime": "Ardisa native execution"}
```
The first request may take longer while Cargo builds the CLI. The demo binds to localhost and is not a production HTTP server. Stop with Ctrl+C.
