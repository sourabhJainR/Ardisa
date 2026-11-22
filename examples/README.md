# Ardisa application showcase

These examples demonstrate Ardisa as the owner of application logic, using the current compiler/runtime.

| Example | Ardisa source | Host integration | What it demonstrates |
|---|---|---|---|
| Console | console.ardisa | Ardisa CLI | Direct native execution and terminal output |
| Desktop | desktop_app.ardisa | Python Tkinter | Ardisa logic displayed in a desktop window |
| Web | web_app.ardisa | Python standard-library HTTP server + HTML | Ardisa logic invoked through a local HTTP endpoint |

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
python examples/desktop-app/app.py
```

## Run the web application

See web-app/README.md.

```sh
python examples/web-app/server.py
```

Open http://127.0.0.1:8080 and select Run check again to invoke Ardisa again. Stop the local server with Ctrl+C.

## Current platform boundary

The .ardisa files own the application behaviour and are checked/executed by Ardisa's compiler and native runtime. Desktop windowing, HTTP serving and browser rendering are host integrations because Ardisa does not yet provide native GUI, HTTP server, or browser/WASM APIs. The adapters are deliberately small and contain no business rules. These samples are local learning demos, not production deployment templates.
