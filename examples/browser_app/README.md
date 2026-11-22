# Ardisa browser runtime example

This example runs an ARDISA-EXEC-V1 application artifact inside the Ardisa
WebAssembly runtime. The browser JavaScript is a thin host adapter; the
application function remains Ardisa source.

From the repository root:

```sh
cargo build -p ardisa --release
./target/release/ardisa bootstrap compile examples/browser_app.ardisa examples/browser_app/app.aexe
wasm-pack build crates/ardisa-wasm --target web --out-dir ../../examples/browser_app/pkg
cd examples/browser_app
python3 -m http.server 8000
```

Open `http://127.0.0.1:8000`. Serve over HTTP rather than opening the HTML file
directly so browser module loading and fetch work.

The v1 browser ABI accepts at most 64 string arguments and returns a string.
It enforces artifact, argument, instruction, call-depth and result limits.
It exposes no ambient DOM, network, filesystem, process or clock capability.
The example's JavaScript uses safe DOM text APIs and does not evaluate source
or artifact contents.

This is an interpreted Ardisa executable running in a WebAssembly-compiled
runtime. It is not yet a direct per-application Ardisa-to-WASM code generator.
