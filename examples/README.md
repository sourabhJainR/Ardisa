# Ardisa application showcase

These examples demonstrate Ardisa as the owner of application logic, using the current compiler/runtime.

| Example | Ardisa source | Host integration | What it demonstrates |
|---|---|---|---|
| Console | console.ardisa | Ardisa CLI | Direct native execution and terminal output |
| Desktop | desktop_app.ardisa | Python Tkinter | Ardisa logic displayed in a desktop window |
| Web | web_app.ardisa | Python standard-library HTTP server + HTML | Ardisa logic invoked through a local HTTP endpoint |
| Native HTTP | shared_app_logic.ardisa | Ardisa CLI native HTTP host | Native HTTP request routed into a compiled Ardisa function |

## Download and build

Install Rust stable with Cargo, then:

```sh
git clone https://github.com/sourabhJainR/Ardisa.git
cd Ardisa
cargo build --release -p ardisa
cargo test --workspace --all-targets
```

Optional: install the CLI into Cargo's user binary directory:

```sh
cargo install --path crates/aifocus-cli --force
```

## Run the console application

```sh
cargo run -p ardisa -- check examples/console.ardisa
cargo run -p ardisa -- run examples/console.ardisa
```

Expected output:

```text
Hello from Ardisa
```

## Run the desktop application

See desktop-app/README.md.

```sh
python3 examples/desktop-app/app.py  # macOS/Linux; use py -3 on Windows
```

## Run the web application

See web-app/README.md.

```sh
python3 examples/web-app/server.py  # macOS/Linux; use py -3 on Windows
```

Open http://127.0.0.1:8080 and select Run check again to invoke Ardisa again. Stop the local server with Ctrl+C.

## Run the native Ardisa HTTP application

The native HTTP host compiles the shared Ardisa handler and runs it through the native VM. It accepts HTTP/1.0 and HTTP/1.1 with bounded Content-Length bodies. It rejects chunked transfer encoding, duplicate Content-Length, unsupported Expect framing and unsafe request paths.

```sh
cargo run -p ardisa -- serve examples/shared_app_logic.ardisa --allow-http-listen 127.0.0.1:8080 GET / home
```

Open http://127.0.0.1:8080. The example handler ABI is:

```ardisa
fn home(method: String, path: String, query: String, body: String) -> String
  "Hello from shared Ardisa application logic"
```

The explicit `--allow-http-listen` flag grants the listener capability. The server handles one connection at a time and has no TLS support; bind to loopback for local development. Do not expose this initial host directly to the public internet.

## Current platform boundary

The .ardisa files own the application behaviour and are checked/executed by Ardisa's compiler and native runtime. Desktop windowing, The native HTTP listener is an initial bounded host; it is not a production multi-connection/TLS server. Desktop windowing and browser rendering remain host integrations because Ardisa does not yet provide a concrete native GUI backend or browser/WASM target. The adapters are deliberately small and contain no business rules. These samples are local learning demos, not production deployment templates.
